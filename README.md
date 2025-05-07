# FRONTEND

- Frontend I use framework Flutter and code Dart. You can download IMAGES fischerscode/flutter:latest and install a container to run frontend

- You can find more information to set up environment flutter on link: https://docs.flutter.dev/

1. Command to run frontend

```bash
flutter run -d web-server --web-port=yourport
```

# BACKEND

- I use GhostFaceNets model to get depth map. You can can read the way to install, train and eval in https://github.com/HamadYA/GhostFaceNets

* You also need to install Uvicorn library to host backend

2. Command to run backend:

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn main:app --host 0.0.0.0 --port yourport --reload
```

# DEMO

- You can watch video demo in UIT AICLUB facebook page, link: https://www.facebook.com/share/v/16RkW6rHif/
