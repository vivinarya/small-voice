# scripts/generate_pdf.py
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def build_pdf():
    pdf_filename = "Reachy_Mini_Integration_Setup_Guide.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#334155'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=8
    )
    
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F8FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    story = []
    
    # Header block
    story.append(Paragraph("Reachy Mini Integration & Setup Guide", title_style))
    story.append(Paragraph("<b>Version:</b> 1.0 | <b>Release Date:</b> July 22, 2026", body_style))
    story.append(Paragraph("<b>Author:</b> Antigravity Pair Programmer", body_style))
    story.append(Spacer(1, 15))
    
    # Horizontal rule
    hr = Table([['']], colWidths=[504])
    hr.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(hr)
    story.append(Spacer(1, 15))
    
    # Section 1
    story.append(Paragraph("1. System Overview & Architecture", h1_style))
    story.append(Paragraph(
        "This system integrates the offline pluggable edge voice assistant (Baymax) running on the <b>Jetson Orin Nano</b> "
        "with the physical <b>Reachy Mini robot</b> controlled by a <b>Raspberry Pi Zero 2W</b>. "
        "The voice assistant features hybrid offline/online LLM inference, local text-to-speech (Piper), speech-to-text (Whisper), "
        "and local knowledge vault (NCERT textbooks / NPS expo info) grounding with DuckDuckGo fallback.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Hardware Communication Ports:</b>", body_style
    ))
    story.append(Paragraph("• <b>Port 5001 (TCP):</b> Motor Server on Pi Zero (receives joint movement commands and blink configurations).", bullet_style))
    story.append(Paragraph("• <b>Port 5002 (TCP):</b> Microphone Server on Pi Zero (streams raw 16kHz audio from physical robot mic).", bullet_style))
    story.append(Paragraph("• <b>Port 5003 (TCP):</b> Speaker Server on Pi Zero (receives raw WAV audio streams to play through robot speakers).", bullet_style))
    story.append(Paragraph("• <b>Port 5000 (TCP):</b> Camera Server on Pi Zero (streams H264 video feed via rpicam-vid to Jetson).", bullet_style))
    story.append(Spacer(1, 10))
    
    # Section 2
    story.append(Paragraph("2. Integrated Code Base Changes", h1_style))
    
    story.append(Paragraph("A. Robot Codebase (reachy_mini_custom)", h2_style))
    story.append(Paragraph(
        "Modified <code>reachy_mini_v6/pi_robot/robot_server.py</code> to execute the <code>rpicam-vid</code> "
        "camera streaming subprocess automatically in the background in Python when the server starts, eliminating the need to run <code>run.sh</code> manually:",
        body_style
    ))
    
    code_pi = (
        "def start_camera_stream():\n"
        "    def run_camera():\n"
        "        while True:\n"
        "            try:\n"
        "                proc = subprocess.Popen([\n"
        "                    'rpicam-vid', '-t', '0', '--inline', '--width', '640', '--height', '480',\n"
        "                    '--framerate', '30', '--codec', 'h264', '--listen', '-o', 'tcp://0.0.0.0:5000'\n"
        "                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "                proc.wait()\n"
        "            except Exception as e:\n"
        "                print(f\"Camera stream error/not installed: {e}\")\n"
        "            time.sleep(0.5)\n"
        "    threading.Thread(target=run_camera, daemon=True).start()\n\n"
        "start_camera_stream()"
    )
    story.append(Paragraph(code_pi.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))
    
    story.append(Paragraph("B. Assistant Codebase (small-voice-main)", h2_style))
    story.append(Paragraph(
        "• <b>config.yaml & config-orin.yaml:</b> Appended robot configuration parameters:<br/>"
        "<code>robot:<br/>&nbsp;&nbsp;enabled: false<br/>&nbsp;&nbsp;ip: \"\"</code>",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>src/config.py:</b> Added <code>robot_enabled</code> (bool) and <code>robot_ip</code> (str) to AppConfig with default values for backward compatibility.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>src/audio/robot.py (New File):</b> Created `RobotController` that handles socket connections to the Pi Zero. "
        "It plays synthesized WAV audio on port 5003 and performs natural neck sway motor gestures on port 5001 during speech.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>src/main.py:</b> Integrated `RobotController`. If <code>robot.enabled</code> is set to <code>true</code>, the local audio stream is bypassed and sent directly to the robot speaker server.",
        bullet_style
    ))
    
    story.append(PageBreak())
    
    # Section 3
    story.append(Paragraph("3. Wake Word Training & Integration", h1_style))
    story.append(Paragraph(
        "To replace the old Jarvis wake word with <b>Baymax</b>, a custom classification model was trained and integrated:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>scripts/train_wakeword.py (New File):</b> Automates training. Synthesizes positive clips ('Hey Baymax', 'Baymax') and negative clips ('Hey Jarvis', 'Alexa', 'Siri', etc.) using your local Piper TTS engine, varies speeds and volume, extracts features, trains a PyTorch Classifier, and exports the final model to ONNX.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>assets/wakeword_models/hey_baymax_v0.1.onnx & .data (New Files):</b> Pre-trained custom ONNX wake word classification model and weight files, generated by the training script.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>src/audio/wakeword.py:</b> Updated default wake word model path to the custom <code>hey_baymax_v0.1.onnx</code> model and set detection sensitivity threshold to <code>0.2</code>.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>src/main.py:</b> Updated local mic loop initializer to load the <code>hey_baymax_v0.1.onnx</code> model.",
        bullet_style
    ))
    story.append(Spacer(1, 10))
    
    # Section 4
    story.append(Paragraph("4. Step-by-Step Setup & Execution Guide", h1_style))
    story.append(Paragraph(
        "Follow these steps to run the complete setup from scratch:",
        body_style
    ))
    
    story.append(Paragraph("Step 1: Start the Robot Server (on the Pi Zero 2W)", h2_style))
    story.append(Paragraph(
        "1. SSH into the Pi Zero 2W.<br/>"
        "2. Navigate to the Pi robot code directory:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>cd /reachy_mini/Reachy_mini_custom/reachy_mini_v6/pi_robot</code><br/>"
        "3. Start the server:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>python3 robot_server.py</code>",
        body_style
    ))
    
    story.append(Paragraph("Step 2: Configure the Assistant (on Jetson Orin Nano / Windows)", h2_style))
    story.append(Paragraph(
        "1. Open <code>config.yaml</code> (or copy <code>config-orin.yaml</code> to <code>config.yaml</code> on the Jetson).<br/>"
        "2. Enable the robot connection and enter the Pi Zero's IP address:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>robot:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;enabled: true<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ip: \"192.168.1.XX\" # <-- Put your Pi Zero's IP here</code>",
        body_style
    ))
    
    story.append(Paragraph("Step 3: Run the Main Assistant loop", h2_style))
    story.append(Paragraph(
        "1. Open a terminal in the <code>small-voice-main</code> directory.<br/>"
        "2. Activate your virtual environment (e.g. <code>source jarvis-env/bin/activate</code> on Jetson).<br/>"
        "3. Execute the assistant:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>python3 src/main.py</code>",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated PDF: {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
