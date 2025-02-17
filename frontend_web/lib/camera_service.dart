import 'dart:async';
import 'dart:convert';
import 'dart:html' as html;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'dart:js' as js;

class CameraService extends ChangeNotifier {
  html.VideoElement? _videoElement;
  html.MediaStream? _mediaStream;
  Timer? _captureTimer;
  bool _cameraAvailable = false;
  bool _isCameraStopped = true;
  String _errorMessage = '';
  String _check_double_result = '';
  int _count_noone = 0;
  String detectionResult = '';
  String resultText = '';
  bool _isPaused = false;
  final FlutterTts flutterTts = FlutterTts();

  bool get cameraAvailable => _cameraAvailable;
  bool get isCameraStopped => _isCameraStopped;



  set cameraAvailable(bool value) {
    _cameraAvailable = value;
    notifyListeners();
  }

  set isCameraStopped(bool value) {
    _isCameraStopped = value;
    notifyListeners();
  }

  // Add methods to control capture
  void pauseCapture() {
    _isPaused = true;
    _captureTimer?.cancel();
    // Don't pause the video stream itself
    notifyListeners();
  }

  Future<void> _reinitializeStream() async {
    try {
      if (_mediaStream != null) {
        // Stop existing tracks
        _mediaStream!.getTracks().forEach((track) => track.stop());
      }

      // Get new media stream
      final mediaDevices = html.window.navigator.mediaDevices;
      if (mediaDevices != null) {
        final newStream = await mediaDevices.getUserMedia({
          'video': {
            'width': 1280,
            'height': 720,
          }
        });

        _mediaStream = newStream;
        if (_videoElement != null) {
          _videoElement!.srcObject = newStream;
          await _videoElement!.play();
        }

        cameraAvailable = true;
        isCameraStopped = false;
        _startPeriodicCapture();
        notifyListeners();
      }
    } catch (e) {
      print('Error reinitializing stream: $e');
      cameraAvailable = false;
      notifyListeners();
    }
  }
  
  void resumeCapture() {
    if (_isPaused) {
      _isPaused = false;
      
      if (detectionResult.isNotEmpty) {
        detectionResult = '';
        notifyListeners();
      }

      // Check if video element and stream are still valid
      if (_videoElement != null && _mediaStream != null) {
        // Ensure video is playing
        _videoElement!.play().then((_) {
          if (!_isPaused && !isCameraStopped) {
            Future.delayed(Duration(milliseconds: 500), () {
              _startPeriodicCapture();
              notifyListeners();
            });
          }
        }).catchError((error) {
          print('Error resuming video: $error');
          // If there's an error, try to reinitialize the stream
          _reinitializeStream();
        });
      } else {
        // If video element or stream is invalid, reinitialize
        _reinitializeStream();
      }
    }
  }

  /// Initialize camera with Chrome 86 compatibility
  Future<void> initializeCamera() async {
    print("Starting camera initialization");
    try {
      // Stop any existing stream
      if (_mediaStream != null) {
        _mediaStream!.getTracks().forEach((track) => track.stop());
      }

      if (js.context['navigator']['mediaDevices'] != null) {
        await _initializeModernCamera();
      } else {
        await _initializeLegacyCamera();
      }

      cameraAvailable = true;
      isCameraStopped = false;
      _isPaused = false;
      _errorMessage = '';
      
      print("Camera initialized successfully");
      _startPeriodicCapture();
      
    } catch (e) {
      _handleError('Failed to initialize camera: $e');
      print('Camera initialization error details: $e');
    }
  }
  
  Future<void> _initializeModernCamera() async {
    final mediaDevices = html.window.navigator.mediaDevices;
    if (mediaDevices == null) {
      throw Exception('Media devices not available');
    }

    // Stop any existing stream
    if (_mediaStream != null) {
      _mediaStream!.getTracks().forEach((track) => track.stop());
    }

    try {
      final mediaStream = await mediaDevices.getUserMedia({
        'video': {
          'width': 1280,
          'height': 720,
        }
      });

      _setupStream(mediaStream);
    } catch (e) {
      print('Modern camera initialization failed, trying legacy method: $e');
      await _initializeLegacyCamera();
    }
  }

  Future<void> _initializeLegacyCamera() async {
    try {
      // Create a video element to receive the stream
      final videoElement = html.VideoElement();
      
      // Set up promise-based getUserMedia call using JavaScript
      final navigator = js.context['navigator'];
      
      final constraints = js.JsObject.jsify({
        'video': {
          'mandatory': {
            'minWidth': '1280',
            'minHeight': '720'
          }
        }
      });

      final completer = Completer<html.MediaStream>();

      // Define success callback
      js.context['handleSuccess'] = js.allowInterop((dynamic jsStream) {
        // Create a new video element
        videoElement.srcObject = jsStream;
        
        // Convert the JavaScript MediaStream to a Dart MediaStream
        final dartStream = html.MediaStream(jsStream);
        completer.complete(dartStream);
      });

      // Define error callback
      js.context['handleError'] = js.allowInterop((dynamic error) {
        completer.completeError('Failed to get user media: ${error.toString()}');
      });

      // Call getUserMedia with our callbacks
      js.context.callMethod('eval', ['''
        navigator.getUserMedia = (
          navigator.getUserMedia ||
          navigator.webkitGetUserMedia ||
          navigator.mozGetUserMedia ||
          navigator.msGetUserMedia
        );
        
        if (navigator.getUserMedia) {
          navigator.getUserMedia(
            ${js.context['JSON'].callMethod('stringify', [constraints])},
            handleSuccess,
            handleError
          );
        } else {
          handleError("getUserMedia not supported");
        }
      ''']);

      final mediaStream = await completer.future;
      _setupStream(mediaStream);

    } catch (e) {
      throw Exception('Failed to initialize legacy camera: $e');
    }
  }

  void _setupStream(html.MediaStream mediaStream) {
    _mediaStream = mediaStream;
    _setupVideoElement(mediaStream);
    _playVideo();
    _ensureVideoReady();

    cameraAvailable = true;
    isCameraStopped = false;
    _errorMessage = '';
    print("Camera initialized successfully");

    _startPeriodicCapture();
  }

  /// set up video element
  void _setupVideoElement(html.MediaStream stream) {
    _videoElement = html.VideoElement()
      ..srcObject = stream
      ..autoplay = true
      // Use setAttribute instead of direct property assignment
      ..setAttribute('playsinline', 'true');

    _videoElement!.style
      ..setProperty('width', '100%')
      ..setProperty('height', '100%')
      ..setProperty('-webkit-transform', 'scaleX(-1)')
      ..setProperty('transform', 'scaleX(-1)')
      ..setProperty('object-fit', 'cover');

    // Add error handling for video element
    _videoElement!.onError.listen((event) {
      print('Video element error: ${event.toString()}');
      _reinitializeStream();
    });

    ui.platformViewRegistry.registerViewFactory(
      'videoElement',
      (int viewId) => _videoElement!,
    );
  }

  Future<void> _playVideo() async {
    try {
      await _videoElement!.play();
    } catch (e) {
      throw Exception("Error playing video: $e");
    }
  }

  Future<void> _ensureVideoReady() async {
    if (_videoElement == null) return;
    
    if (_videoElement!.videoWidth > 0 && _videoElement!.videoHeight > 0) return;

    await Future.any([
      _videoElement!.onLoadedMetadata.first,
      Future.delayed(const Duration(seconds: 3)),
    ]);

    if (_videoElement!.videoWidth == 0 || _videoElement!.videoHeight == 0) {
      _videoElement!.width = 640;
      _videoElement!.height = 480;
    }
  }

  void _startPeriodicCapture() {
    if (_isPaused || !cameraAvailable || isCameraStopped) return;

    _captureTimer?.cancel();
    _captureTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (cameraAvailable && !isCameraStopped && !_isPaused) {
        captureAndQuery();
      }
    });

    // Only do initial capture if not paused and camera is available
    if (!_isPaused && cameraAvailable && !isCameraStopped) {
      captureAndQuery();
    }
  }

  Future<String> captureAndQuery() async {
    if (!cameraAvailable || _videoElement == null || _isPaused) {
      return 'Camera not available or paused';
    }

    try {
      final canvas = html.CanvasElement(
        width: _videoElement!.videoWidth,
        height: _videoElement!.videoHeight,
      );
      
      // Ensure video element is ready
      if (_videoElement!.videoWidth == 0 || _videoElement!.videoHeight == 0) {
        return 'Video element not ready';
      }

      canvas.context2D.drawImage(_videoElement!, 0, 0);


      final dataUrl = canvas.toDataUrl('image/jpeg');
      final base64Image = dataUrl.split(',')[1];

      final response = await http
          .post(
            Uri.parse('https://aiclub.uit.edu.vn/gpu150/ohmni_be/detect'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'image': base64Image}),
          )
          .timeout(const Duration(seconds: 100));

      if (response.statusCode == 200) {
        final jsondecodebody = jsonDecode(utf8.decode(response.bodyBytes));

        if (jsondecodebody['result_image'] is List) {
          List<String> resultList = List<String>.from(jsondecodebody['result_image'])
              .where((item) => item != "Unknown")
              .toList();

          resultList.sort();
          String resultText = resultList.join(", ");
          
          if (resultText.isEmpty) {
            detectionResult = '';
          } else {
            if (!_isPaused) {   
              detectionResult = "Xin chào $resultText tôi có thể giúp bạn thông qua chat bot bên cạnh";
              _speak("Xin chào $resultText, tôi có thể giúp bạn thông qua chát bót");
              notifyListeners();
            }
          }
          
          return detectionResult;
        }
      }
      
      return 'Error processing response';
    } catch (e) {
      print('Error in captureAndQuery: $e');
      return 'Error processing image: $e';
    }
  }

  /// handle bug và log
  void _handleError(String message) {
    _errorMessage = message;
    cameraAvailable = false;
    print(message);
  }

  /// speak
  Future<void> _speak(String text) async {
    await flutterTts.setLanguage("vi-VN");
    await flutterTts.setPitch(1.0);
    await flutterTts.speak(text);
  }
}
