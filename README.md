# Github
[https://github.com/KonstFed/image-realness](https://github.com/KonstFed/image-realness)

# Innobots

This project is presented as solution to hackaton `InnoGlobalHack`.

# How to run

You need to create folder weights and download weights there.

Overall draft of dependecies can be found `requirements.txt`. Highly recommend to create virtual environment for testing repository

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Literal requirements can be found as `freeze_requirements.txt`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r freeze_requirements.txt
```

Also you can try installing it via docker (Doo not forget to download weights)

```bash
docker build . -t innobot-backend
docker run -p 5000:5000 -d innobot-backend
```

# BOT & CLI

You can use our easy bot located in `bot/simple_bot.py`. To do so install one library
[aiogram](https://aiogram.dev/). Then create `secret.json` in the same directory similar to `example_secret.json`

You can use simple CLI. Just type:
```bash
python3 cli.py path-to-folder
```
`path-to-folder` is path where all needed for test data contained