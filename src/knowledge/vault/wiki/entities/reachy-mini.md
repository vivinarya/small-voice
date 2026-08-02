---
entity: reachy-mini
type: robot
---
# Reachy Mini — AI Robot at NPS ITPL

Reachy Mini is a compact, expressive humanoid robot developed by Pollen Robotics, used as the physical embodiment of the Baymax AI voice assistant at NPS ITPL. It was demonstrated at the school's Robotics Expo on January 31, 2026.

## About the Robot
- Robot Name: Reachy Mini
- Manufacturer: Pollen Robotics (France)
- Type: Compact humanoid robot with expressive head and arm movements
- AI System: Baymax Edge Voice Assistant (running locally on NVIDIA Jetson Orin)

## Role at NPS ITPL
The Reachy Mini robot was deployed at NPS ITPL as part of a student-led AI project. It serves as a school information assistant, answering questions about the school, curriculum, events, and academic topics. The robot was showcased at the Robotics Expo held on January 31, 2026.

## Technical Setup
The Reachy Mini at NPS ITPL runs:
- A fully local AI pipeline with no cloud dependency
- Speech-to-Text using Whisper (FasterWhisper)
- Language Model using Ollama with Qwen 2.5 1.5B parameter model
- Text-to-Speech using Piper TTS
- Knowledge retrieval from school documents and a wiki knowledge base
- Wake word detection (say "Baymax" to activate)
- The entire system runs on an NVIDIA Jetson Orin Developer Kit

## Capabilities
The Reachy Mini AI assistant can:
- Answer questions about NPS ITPL school information
- Explain academic subjects from uploaded textbooks
- Provide information about school events like HackNexus 2026 and the Robotics Expo
- Nod and animate expressively while speaking
- Be activated by voice or through a web chat interface

## Student Project Team
This robot AI system was built by students of NPS ITPL as a project showcasing applied AI, robotics, and embedded systems development.
