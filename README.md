# Instructions

## For MALI

open docker desktop if not already running

docker build -t joshuamali/cricket-clips:latest .

docker push joshuamali/cricket-clips:latest

## For Liam

Open Command Prompt (Win + R -> "cmd" -> enter)

docker pull joshuamali/cricket-clips:latest

docker run --rm -p 8501:8501 joshuamali/cricket-clips:latest

Open in browser -> http://localhost:8501.

TO CLOSE: In command prompt -> ctrl + c