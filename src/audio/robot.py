# src/audio/robot.py
import socket
import threading
import time
import random
import logging

logger = logging.getLogger(__name__)

class RobotController:
    def __init__(self, ip: str = "", enabled: bool = False):
        self.ip = ip
        self.enabled = enabled and bool(ip)
        self.cmd_sock = None
        self.speaker_sock = None
        self.is_talking = False
        self.talking_thread = None
        
        if self.enabled:
            # Connect in a background thread to avoid blocking server startup
            threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        print(f"[Robot] Connecting to robot at {self.ip}...", flush=True)
        try:
            self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_sock.settimeout(3.0)
            self.cmd_sock.connect((self.ip, 5001))
            
            self.speaker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.speaker_sock.settimeout(3.0)
            self.speaker_sock.connect((self.ip, 5003))
            
            print(f"[Robot] Successfully connected to ports 5001 and 5003 on {self.ip}!", flush=True)
            # Send initial configuration
            self.send_command("BLINK:ON|")
            self.go_home()
        except Exception as e:
            print(f"[Robot Error] Failed to connect to robot: {e}", flush=True)
            self.cmd_sock = None
            self.speaker_sock = None

    def send_command(self, cmd: str):
        if self.cmd_sock:
            try:
                self.cmd_sock.sendall(cmd.encode('utf-8'))
            except Exception:
                pass

    def go_home(self):
        # Home positions for SC15 servos (1-6) and other features
        # 1-6 are the neck parallel mechanism joints. 7 is base pan, 9,10 are eyes.
        # Home positions: Base 500, Eyes 200, 225. Speed 150.
        self.send_command("1,575,150 2,470,150 3,560,150 4,460,150 5,575,150 6,485,150 7,500,150 9,200,150 10,225,150|")

    def play_audio(self, wav_bytes: bytes):
        if self.speaker_sock and wav_bytes:
            try:
                self.is_talking = True
                self.start_talking_loop()
                header = f"{len(wav_bytes):<10}".encode('utf-8')
                self.speaker_sock.sendall(header + wav_bytes)
                # Calculate duration of the wav file to wait
                # 1 channel, 16-bit (2 bytes per sample), 22050Hz
                # wav file has a 44-byte header
                pcm_len = len(wav_bytes) - 44
                duration = pcm_len / (22050.0 * 2.0)
                time.sleep(duration)
                self.is_talking = False
            except Exception as e:
                print(f"[Robot Speaker Error] {e}", flush=True)
                self.is_talking = False

    def start_talking_loop(self):
        if self.talking_thread and self.talking_thread.is_alive():
            return
        
        def loop():
            # Send simple neck movement commands to simulate speaking/gesturing
            # We sway around the home positions
            while self.is_talking:
                # Randomize small neck movements
                # Home positions: 1:575, 2:470, 3:560, 4:460, 5:575, 6:485
                j1 = 575 + random.randint(-40, 40)
                j2 = 470 + random.randint(-40, 40)
                j3 = 560 + random.randint(-40, 40)
                j4 = 460 + random.randint(-40, 40)
                j5 = 575 + random.randint(-40, 40)
                j6 = 485 + random.randint(-40, 40)
                
                # Base pan: home is 500
                b = 500 + random.randint(-80, 80)
                
                cmd = f"1,{j1},300 2,{j2},300 3,{j3},300 4,{j4},300 5,{j5},300 6,{j6},300 7,{b},300|"
                self.send_command(cmd)
                time.sleep(0.8)
            self.go_home()

        self.talking_thread = threading.Thread(target=loop, daemon=True)
        self.talking_thread.start()
