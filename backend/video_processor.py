import os
import json
import shutil
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.update import Updater
from yt_dlp.version import __version__ as ytdlp_version

from .media_tools import ffmpeg_cmd, ffprobe_cmd, get_ffmpeg_bin

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器，使用yt-dlp下载和转换视频"""
    
    def __init__(self):
        self._default_user_agent = os.getenv(
            "YDL_USER_AGENT",
            # 使用常见的桌面浏览器 UA，避免被判定为机器人后限速或截断
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        )
        format_value = os.getenv(
            "YDL_FORMAT",
            # 优先下载 m4a 格式，避免 webm 等需要转换的格式
            "bestaudio[ext=m4a]/bestaudio/best",
        )
        self.ydl_opts = {
            'format': format_value,  # 优先下载最佳音频轨，可用 YDL_FORMAT 覆盖
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # 直接在提取阶段转换为单声道 16k（空间小且稳定）
                'preferredcodec': 'm4a',
                'preferredquality': '64'
            }],
            # 全局FFmpeg参数：单声道 + 16k 采样率 + 64kbps + faststart
            'postprocessor_args': ['-ac', '1', '-ar', '16000', '-b:a', '64k', '-movflags', '+faststart'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,  # 强制只下载单个视频，不下载播放列表
            'continuedl': True,
            'retries': 10,
            'fragment_retries': 10,
        }

        ffmpeg_bin = get_ffmpeg_bin()
        if ffmpeg_bin:
            self.ydl_opts['ffmpeg_location'] = str(Path(ffmpeg_bin).parent)

        # Support a yt-dlp-specific proxy override in addition to standard
        # HTTP(S)_PROXY env vars that underlying libraries read automatically.
        ytdlp_proxy = os.getenv("YT_DLP_PROXY") or os.getenv("YDL_PROXY")
        if ytdlp_proxy:
            self.ydl_opts['proxy'] = ytdlp_proxy

        # 尝试自动配置 JS 解释器，解决 "No supported JavaScript runtime" 警告
        # yt-dlp 默认使用 deno，也支持 node
        js_interpreter = os.getenv("YDL_JS_INTERPRETER")
        js_runtime_name = None
        if not js_interpreter:
            # 优先检查 deno（yt-dlp 默认），其次是 node
            deno_path = shutil.which('deno')
            node_path = shutil.which('node') or shutil.which('nodejs')
            if deno_path:
                js_interpreter = deno_path
                js_runtime_name = 'deno'
            elif node_path:
                js_interpreter = node_path
                js_runtime_name = 'node'
        else:
            # 用户指定了 JS 解释器路径，需要确定运行时名称
            if 'deno' in js_interpreter.lower():
                js_runtime_name = 'deno'
            else:
                js_runtime_name = 'node'

        if js_interpreter and js_runtime_name:
            # Python API 使用字典格式: {runtime: {config}}
            self.ydl_opts['js_runtimes'] = {js_runtime_name: {'path': js_interpreter}}
            logger.debug(f"已配置 yt-dlp 使用 JS 运行时: {js_runtime_name} (path={js_interpreter})")

        # 始终启用远程组件下载，解决 YouTube JS challenge
        # 即使没有找到本地 JS 运行时，也设置此选项，让 yt-dlp 尝试使用默认的 deno
        # 参见: https://github.com/yt-dlp/yt-dlp/wiki/EJS
        # Python API 格式: set of strings，如 {'ejs:github'}
        self.ydl_opts['remote_components'] = {'ejs:github'}

        # 注意：不在这里设置默认 cookies，而是在 download_and_convert 中根据 URL 选择
        # YDL_COOKIEFILE 用于 YouTube，BILIBILI_COOKIE_FILE 用于 Bilibili

        extractor_args_json = os.getenv("YDL_EXTRACTOR_ARGS_JSON")
        if extractor_args_json:
            try:
                extractor_args = json.loads(extractor_args_json)
                if isinstance(extractor_args, dict):
                    self.ydl_opts['extractor_args'] = extractor_args
                else:
                    logger.warning("YDL_EXTRACTOR_ARGS_JSON 需要是 JSON 对象: %s", extractor_args_json)
            except json.JSONDecodeError:
                logger.warning("无法解析 YDL_EXTRACTOR_ARGS_JSON: %s", extractor_args_json)
        else:
            default_player_client = os.getenv("YDL_DEFAULT_PLAYER_CLIENT")
            if default_player_client:
                clients = [item.strip() for item in default_player_client.split(',') if item.strip()]
                if clients:
                    self.ydl_opts['extractor_args'] = {'youtube': {'player_client': clients}}
            # 注意：不设置默认 player_client，因为 android client 不支持 cookies
            # 让 yt-dlp 自动选择合适的 client，配合 remote_components 解决 JS challenge
        chunk_size = os.getenv("YDL_HTTP_CHUNK_SIZE")
        if chunk_size:
            try:
                self.ydl_opts['http_chunk_size'] = int(chunk_size)
            except ValueError:
                logger.warning("YDL_HTTP_CHUNK_SIZE 非法: %s", chunk_size)
        format_candidates = os.getenv("YDL_FORMAT_MAX_CANDIDATES")
        if format_candidates:
            try:
                self._format_max_candidates = max(1, int(format_candidates))
            except ValueError:
                logger.warning("YDL_FORMAT_MAX_CANDIDATES 非法: %s", format_candidates)
                self._format_max_candidates = 20
        else:
            self._format_max_candidates = 20
        self._update_hint_checked = False
        self._cached_update_hint: Optional[str] = None

    async def download_and_convert(self, url: str, output_dir: Path, *, video_info: Optional[dict] = None) -> tuple[str, str]:
        """
        下载视频并转换为m4a格式

        Args:
            url: 视频链接
            output_dir: 输出目录
            video_info: 预获取的视频元数据（可选），避免重复调用 yt-dlp

        Returns:
            转换后的音频文件路径和视频标题
        """
        try:
            # 创建输出目录
            output_dir.mkdir(exist_ok=True)
            
            # 生成唯一的文件名
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
            # 更新yt-dlp选项
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = output_template

            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            if hostname.endswith("bilibili.com"):
                # 哔哩哔哩对 Referer/UA 较为敏感，缺失时常出现下载被截断
                headers = {
                    'Referer': 'https://www.bilibili.com/',
                    'User-Agent': self._default_user_agent,
                }
                existing_headers = dict(ydl_opts.get('http_headers') or {})
                existing_headers.update(headers)
                ydl_opts['http_headers'] = existing_headers

                cookie_file = os.getenv('BILIBILI_COOKIE_FILE')
                if cookie_file:
                    cookie_path = Path(cookie_file).expanduser()
                    if cookie_path.exists():
                        ydl_opts['cookiefile'] = str(cookie_path)
                    else:
                        logger.warning(
                            "BILIBILI_COOKIE_FILE 指定的文件不存在: %s", cookie_path
                        )
            elif hostname.endswith("youtube.com") or hostname.endswith("youtu.be"):
                # YouTube 使用专门的 cookies 文件
                cookie_file = os.getenv('YDL_COOKIEFILE')
                if cookie_file:
                    cookie_path = Path(cookie_file).expanduser()
                    if cookie_path.exists():
                        ydl_opts['cookiefile'] = str(cookie_path)
                    else:
                        logger.warning(
                            "YDL_COOKIEFILE 指定的文件不存在: %s", cookie_path
                        )

            logger.info(f"开始下载视频: {url}")
            
            # 直接同步执行，不使用线程池
            # 在FastAPI中，IO密集型操作可以直接await
            candidate_formats: List[str] = []
            probe_info = None
            try:
                probe_info = await asyncio.to_thread(self._probe_video_info, url, ydl_opts)
            except Exception as probe_exc:
                logger.debug(f"探测可用格式失败，使用默认格式重试: {probe_exc}")
            else:
                candidate_formats = self._build_format_candidates(probe_info, ydl_opts.get('format'))

            info = None
            last_exc: Optional[Exception] = None
            if candidate_formats:
                for index, format_id in enumerate(candidate_formats):
                    trial_opts = ydl_opts.copy()
                    if format_id:
                        trial_opts['format'] = format_id
                    else:
                        trial_opts.pop('format', None)

                    log_label = format_id or ydl_opts.get('format') or '<auto>'
                    if index == 0:
                        logger.info(f"尝试下载格式: {log_label}")
                    else:
                        logger.info(f"尝试回退格式: {log_label}")

                    try:
                        info = await self._run_ytdlp(url, trial_opts)
                        if index > 0:
                            logger.info(f"使用回退格式下载成功: {log_label}")
                        break
                    except yt_dlp.utils.DownloadError as exc:
                        last_exc = exc
                        if not self._should_retry_format(exc):
                            raise
                        logger.warning(f"格式 {log_label} 下载失败（{exc}），尝试下一档格式…")
                        continue
            else:
                try:
                    info = await self._run_ytdlp(url, ydl_opts)
                except yt_dlp.utils.DownloadError as exc:
                    last_exc = exc

            if info is None:
                if last_exc:
                    raise last_exc
                raise Exception("未能获取下载信息")

            # 优先使用缓存的视频信息（如果有的话）
            video_title = info.get('title', 'unknown')
            expected_duration = info.get('duration') or 0
            if video_info:
                video_title = video_info.get('title') or video_title
                expected_duration = video_info.get('duration') or expected_duration
            logger.info(f"视频标题: {video_title}")
            
            # 查找生成的m4a文件
            audio_file = str(output_dir / f"audio_{unique_id}.m4a")
            
            if not os.path.exists(audio_file):
                # 如果m4a文件不存在，查找其他音频格式
                for ext in ['webm', 'mp4', 'mp3', 'wav']:
                    potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                else:
                    raise Exception("未找到下载的音频文件")
            
            # 校验时长，如果和源视频差异较大，尝试一次ffmpeg规范化重封装
            try:
                def _probe(path: str) -> float:
                    cmd = ffprobe_cmd(
                        '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        path
                    )
                    out_local = subprocess.check_output(cmd).decode().strip()
                    return float(out_local) if out_local else 0.0

                out = _probe(audio_file)
                actual_duration = float(out) if out else 0.0
            except Exception as _:
                actual_duration = 0.0

            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"音频时长异常，期望{expected_duration}s，实际{actual_duration}s，尝试重封装修复…"
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                    fix_cmd = ffmpeg_cmd(
                        '-y', '-i', audio_file,
                        '-vn', '-c:a', 'aac', '-b:a', '160k',
                        '-movflags', '+faststart', fixed_path
                    )
                    subprocess.check_call(fix_cmd)
                    # 用修复后的文件替换
                    audio_file = fixed_path
                    # 重新探测
                    actual_duration2 = _probe(audio_file)
                    logger.info(f"重封装完成，新时长≈{actual_duration2:.2f}s")
                except Exception as e:
                    logger.error(f"重封装失败：{e}")
            
            logger.info(f"音频文件已保存: {audio_file}")
            return audio_file, video_title
            
        except Exception as e:
            message = str(e)
            if self._needs_update_hint(e):
                hint = self._get_update_hint()
                if hint and hint not in message:
                    message = f"{message}；{hint}"
                    logger.warning(hint)
            logger.error(f"下载视频失败: {message}")
            raise Exception(f"下载视频失败: {message}") from e

    def get_video_info(self, url: str) -> dict:
        """
        获取视频信息
        
        Args:
            url: 视频链接
            
        Returns:
            视频信息字典
        """
        try:
            opts = {'quiet': True}
            # 继承关键配置，如 JS 运行时、远程组件 和 Cookies
            if 'js_runtimes' in self.ydl_opts:
                opts['js_runtimes'] = self.ydl_opts['js_runtimes']
            if 'remote_components' in self.ydl_opts:
                opts['remote_components'] = self.ydl_opts['remote_components']
            if 'cookiefile' in self.ydl_opts:
                opts['cookiefile'] = self.ydl_opts['cookiefile']
            if 'http_headers' in self.ydl_opts:
                opts['http_headers'] = self.ydl_opts['http_headers']

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}")
            raise Exception(f"获取视频信息失败: {str(e)}")

    async def _run_ytdlp(self, url: str, opts: dict, download: bool = True):
        def _extract():
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=download)

        return await asyncio.to_thread(_extract)

    def _probe_video_info(self, url: str, base_opts: dict) -> dict:
        probe_opts = base_opts.copy()
        # 探测阶段无需下载或执行后处理
        probe_opts.pop('format', None)
        probe_opts.pop('postprocessors', None)
        probe_opts.pop('postprocessor_args', None)
        probe_opts.pop('outtmpl', None)
        probe_opts['noplaylist'] = True
        probe_opts['quiet'] = True
        probe_opts['skip_download'] = True

        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            return ydl.extract_info(url, download=False)

    def _build_format_candidates(self, info: dict, preferred_format: Optional[str]) -> List[str]:
        formats = info.get('formats') or []
        if not formats:
            return self._expand_format_tokens(preferred_format)

        audio_formats = []
        progressive = []

        for fmt in formats:
            format_id = fmt.get('format_id')
            if not format_id:
                continue
            if fmt.get('acodec') in (None, 'none'):
                continue
            if fmt.get('vcodec') in (None, 'none'):
                audio_formats.append(fmt)
            else:
                progressive.append(fmt)

        def _audio_score(fmt: dict) -> tuple:
            abr = fmt.get('abr') or fmt.get('tbr') or 0
            size = fmt.get('filesize') or fmt.get('filesize_approx') or 0
            # 适度偏向 m4a，便于后续音频抽取
            ext_priority = 1 if fmt.get('ext') == 'm4a' else 0
            return (float(abr), ext_priority, float(size))

        audio_formats.sort(key=_audio_score, reverse=True)

        def _progressive_score(fmt: dict) -> tuple:
            height = fmt.get('height') or 0
            bitrate = fmt.get('tbr') or fmt.get('abr') or 0
            return (int(height), float(bitrate))

        progressive.sort(key=_progressive_score, reverse=True)

        candidates: List[str] = []
        preferred_tokens = self._expand_format_tokens(preferred_format)
        available_ids = {
            str(fmt.get('format_id')): True for fmt in formats if fmt.get('format_id')
        }

        for token in preferred_tokens:
            if token in ('bestaudio', 'best', 'worstaudio', 'worst'):
                continue
            if token in available_ids and token not in candidates:
                candidates.append(token)

        for fmt in audio_formats:
            format_id = fmt.get('format_id')
            if not format_id:
                continue
            format_id = str(format_id)
            if format_id not in candidates:
                candidates.append(format_id)

        for fmt in progressive:
            format_id = fmt.get('format_id')
            if not format_id:
                continue
            format_id = str(format_id)
            if format_id not in candidates:
                candidates.append(format_id)

        default_id = info.get('format_id')
        if default_id:
            default_id = str(default_id)
            if default_id not in candidates:
                candidates.append(default_id)

        # 限制尝试数量，避免低质量格式过多导致重复请求
        return candidates[: self._format_max_candidates]

    def _expand_format_tokens(self, format_value: Optional[str]) -> List[str]:
        if not format_value or not isinstance(format_value, str):
            return []
        return [token.strip() for token in format_value.split('/') if token.strip()]

    def _should_retry_format(self, exc: Exception) -> bool:
        text = str(exc).lower()
        retriable_markers = (
            'http error 403',
            'http error 404',
            'http error 410',
            'forbidden',
            'requested format is not available',
        )
        return any(marker in text for marker in retriable_markers)

    def _needs_update_hint(self, exc: Exception) -> bool:
        text = str(exc)
        return 'Requested format is not available' in text

    def _get_update_hint(self) -> Optional[str]:
        if self._update_hint_checked:
            return self._cached_update_hint

        self._update_hint_checked = True
        hint: Optional[str]
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                update_info = Updater(ydl).query_update()
        except Exception as exc:  # pragma: no cover - 网络/权限问题无需打断流程
            logger.debug(f"检查 yt-dlp 更新失败: {exc}")
            hint = None
        else:
            if update_info:
                latest = update_info.version or update_info.tag
                hint = (
                    f"检测到 yt-dlp 有可用更新（最新: {latest}，当前: {ytdlp_version}）。"
                    "请运行 `pip install --upgrade yt-dlp` 后重试。"
                )
            else:
                hint = None

        self._cached_update_hint = hint
        return hint
