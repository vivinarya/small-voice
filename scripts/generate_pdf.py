# scripts/generate_pdf.py
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
    story.append(Paragraph("Reachy Mini Robot & Baymax Setup Guide", title_style))
    story.append(Paragraph("<b>Version:</b> 1.1 | <b>Release Date:</b> July 23, 2026", body_style))
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
    
    # Phase 1
    story.append(Paragraph("Phase 1: Find the Robot's (Pi Zero 2W) IP Address", h1_style))
    story.append(Paragraph(
        "Make sure the robot (Pi Zero 2W) and your Jetson Orin Nano are connected to the same local Wi-Fi network.",
        body_style
    ))
    story.append(Paragraph(
        "1. Open a terminal on your Jetson Orin Nano and run:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>ping -c 3 raspberrypi.local</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>(If you custom-named the robot, try: <code>ping -c 3 reachy.local</code>)</i><br/>"
        "2. Copy the IP address printed in the ping response (e.g., <code>192.168.1.45</code>).<br/>"
        "3. If the ping fails, scan the local network to find the IP:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>arp -a</code>",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Phase 2
    story.append(Paragraph("Phase 2: Start the Robot Server (Raspberry Pi Zero 2W)", h1_style))
    story.append(Paragraph(
        "1. SSH into the Pi Zero 2W using the IP address:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>ssh pi-zero@192.168.1.22</code><br/>"
        "2. Navigate to the Pi robot server directory:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>cd /reachy_mini/Reachy_mini_custom/reachy_mini_v6/pi_robot</code><br/>"
        "3. Start the socket and camera streams:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>python3 robot_server.py</code><br/>"
        "<i>Keep this terminal window running. This initializes the Motor Server (port 5001), Mic Server (port 5002), Speaker Server (port 5003), and background Camera Stream (port 5000).</i>",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Phase 3
    story.append(Paragraph("Phase 3: Set up the Voice Assistant (Jetson Orin Nano)", h1_style))
    story.append(Paragraph(
        "Open a new terminal window on your Jetson Orin Nano.",
        body_style
    ))
    story.append(Paragraph(
        "1. Clone the project and configure the environment:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>git clone https://github.com/vivinarya/small-voice.git</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>cd small-voice</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>python3 -m venv jarvis-env</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>source jarvis-env/bin/activate</code><br/>"
        "2. Install python packages:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>pip install -r requirements-orin.txt</code><br/>"
        "3. Download local Piper TTS assets:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>mkdir -p assets/piper/ assets/piper_voices/</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>tar -xf piper_arm64.tar.gz</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>cp piper/piper assets/piper/</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>chmod +x assets/piper/piper</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>wget -O assets/piper_voices/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>wget -O assets/piper_voices/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json</code><br/>"
        "4. Setup Ollama (Local LLM Engine):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>curl -fsSL https://ollama.com/install.sh | sh</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>ollama pull qwen2.5:3b</code><br/>"
        "5. Update config.yaml with the Robot IP:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>cp config-orin.yaml config.yaml</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Open <code>config.yaml</code> in a text editor (e.g. <code>nano config.yaml</code>) and set:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>robot:</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>&nbsp;&nbsp;enabled: true</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>&nbsp;&nbsp;ip: \"&lt;YOUR_ROBOT_IP&gt;\"</code>",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Phase 4
    story.append(Paragraph("Phase 4: Run the Assistant", h1_style))
    story.append(Paragraph(
        "1. Start the conversational assistant on the Jetson Orin Nano:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>python3 src/main.py</code><br/>"
        "2. Trigger the dialogue by saying **\"Hey Baymax\"** (or **\"Baymax\"**).<br/>"
        "3. Ask your question. Speech synthesized on the Jetson will play directly from the robot speakers (port 5003), and the robot will perform talking sway sways (port 5001) in real-time.",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated PDF: {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
