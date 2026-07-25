# src/audio/robot.py
import threading
import time
import random
import logging
import urllib.request
import json

logger = logging.getLogger(__name__)

WEBSERVER_URL = "http://localhost:8070"


def _http_post(url: str, body: bytes, content_type: str = "application/json", timeout: float = 3.0):
    """Minimal fire-and-forget HTTP POST using stdlib only."""
    try:
        req = urllib.request.Request(
            url,
            method="POST",
            data=body,
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except Exception as e:
        logger.debug("[Robot HTTP] %s → %s", url, e)


class RobotController:
    """
    Routes all robot commands through the local webserver's HTTP API
    (http://localhost:8080) instead of holding raw sockets to the Pi.

    This prevents port contention on Pi ports 5001/5003 between main.py
    and server.py, which both previously tried to open direct connections.

    Startup order:
      1. Pi server (./run.sh on the Pi)
      2. Webserver (uv run server.py on the Jetson)
      3. Browser → enter Pi IP → click Connect
      4. This process (AI model)
    """

    def __init__(self, ip: str = "", enabled: bool = False,
                 webserver_url: str = WEBSERVER_URL):
        self.ip = ip
        self.enabled = enabled and bool(ip)
        self.webserver_url = webserver_url.rstrip("/")
        self.is_talking = False
        self.talking_thread: threading.Thread | None = None

        if self.enabled:
            # Trigger the webserver to connect to the Pi in the background.
            # If the user already connected via the browser UI, the webserver
            # will simply re-confirm the connection (safe to call twice).
            threading.Thread(target=self._connect, daemon=True).start()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        """Ask the webserver to open its sockets to the Pi."""
        print(f"[Robot] Requesting webserver to connect to Pi at {self.ip}...", flush=True)
        body = json.dumps({"ip": self.ip}).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.webserver_url}/api/connect",
                method="POST",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                print(f"[Robot] Webserver connect → {result.get('status', 'ok')}", flush=True)
        except Exception as e:
            print(
                f"[Robot Warning] Could not reach webserver at {self.webserver_url}: {e}\n"
                f"            Make sure the webserver is running and the Pi IP is correct.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Motor commands
    # ------------------------------------------------------------------

    def send_command(self, cmd: str):
        """
        Forward a raw servo command string to the Pi via the webserver.

        Format is the same as before, e.g.:
            "1,575,150 2,470,150 3,560,150 ... 7,500,150 9,200,150 10,225,150|"
        """
        if not self.enabled:
            return
        body = json.dumps({"cmd": cmd}).encode("utf-8")
        _http_post(f"{self.webserver_url}/api/raw_cmd", body)

    def go_home(self):
        """Send all joints to their home positions."""
        self.send_command(
            "1,575,150 2,470,150 3,560,150 4,460,150 "
            "5,575,150 6,485,150 7,500,150 9,200,150 10,225,150|"
        )

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def play_audio(self, wav_bytes: bytes):
        """
        Stream WAV bytes to the Pi speaker via the webserver's /api/play_wav.

        The webserver forwards them over the already-established speaker
        socket using the same length-prefixed framing the Pi expects.
        """
        if not self.enabled or not wav_bytes:
            return
        try:
            self.is_talking = True
            self.start_talking_loop()

            _http_post(
                f"{self.webserver_url}/api/play_wav",
                wav_bytes,
                content_type="application/octet-stream",
                timeout=30.0,   # long timeout — audio can be several seconds
            )

            # Estimate playback duration so we hold is_talking long enough.
            # WAV header = 44 bytes, 1 ch, 16-bit (2 bytes/sample), 22050 Hz.
            pcm_len = max(0, len(wav_bytes) - 44)
            duration = pcm_len / (22050.0 * 2.0)
            time.sleep(duration)

        except Exception as e:
            print(f"[Robot Speaker Error] {e}", flush=True)
        finally:
            self.is_talking = False

    # ------------------------------------------------------------------
    # Talking animation loop
    # ------------------------------------------------------------------

    def start_talking_loop(self):
        """Randomised neck-sway animation while the robot is speaking."""
        if self.talking_thread and self.talking_thread.is_alive():
            return

        def loop():
            while self.is_talking:
                j1 = 575 + random.randint(-40, 40)
                j2 = 470 + random.randint(-40, 40)
                j3 = 560 + random.randint(-40, 40)
                j4 = 460 + random.randint(-40, 40)
                j5 = 575 + random.randint(-40, 40)
                j6 = 485 + random.randint(-40, 40)
                b  = 500 + random.randint(-80, 80)
                cmd = (
                    f"1,{j1},300 2,{j2},300 3,{j3},300 "
                    f"4,{j4},300 5,{j5},300 6,{j6},300 7,{b},300|"
                )
                self.send_command(cmd)
                time.sleep(0.8)
            self.go_home()

        self.talking_thread = threading.Thread(target=loop, daemon=True)
        self.talking_thread.start()
