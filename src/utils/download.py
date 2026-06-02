import subprocess
from pathlib import Path


def download_with_wget(url: str, local_path: Path, user: str, passw: str) -> None:
    """
    Downloads a file using wget with authentication.

    Args:
        url: The source URL.
        local_path: The destination path (Path object or string).
        user: PhysioNet username.
        passw: PhysioNet password.

    Raises:
        subprocess.CalledProcessError: If the download fails.
    """
    # Ensure the directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "wget",
        "-c",  # Continue partial downloads
        "-N",  # Timestamping (only download if newer)
        "-q",  # Quiet mode
        "--user",
        user,
        "--password",
        passw,
        "-O",
        str(local_path),
        url,
    ]

    # Check_call raises CalledProcessError on non-zero exit code
    subprocess.check_call(cmd)
