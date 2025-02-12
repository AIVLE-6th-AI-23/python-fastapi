import mimetypes
import magic

class UnsupportedFileTypeError(Exception):
    pass

TEXT_MIME_TYPES = ["text/plain", "application/json", "application/xml"]
IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp", "image/tiff"]
VIDEO_MIME_TYPES = ["video/mp4", "video/mpeg", "video/avi", "video/quicktime", "video/x-msvideo"]

def get_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if not mime_type:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)

    return mime_type or "application/octet-stream"


def categorize_file(file_path: str) -> str:
    mime_type = get_mime_type(file_path)

    if mime_type in TEXT_MIME_TYPES or mime_type.startswith("text"):
        return "text"
    elif mime_type in IMAGE_MIME_TYPES or mime_type.startswith("image"):
        return "image"
    elif mime_type in VIDEO_MIME_TYPES or mime_type.startswith("video"):
        return "video"
    else:
        raise UnsupportedFileTypeError(f"지원 도지 않는 파일 유형: {mime_type}")
