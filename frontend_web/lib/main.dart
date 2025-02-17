import 'dart:async';
import 'dart:html' as html;
import 'dart:ui' as ui;
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'dart:convert';
import 'chat_widget.dart';
import 'camera_service.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(
    ChangeNotifierProvider(
      create: (context) => CameraService(),
      child: OhmniWelcomeYou(),
    ),
  );
}

class OhmniWelcomeYou extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ohmni',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String _recognitionResult = 'No face recognized';
  String _errorMessage = '';
  FlutterTts flutterTts = FlutterTts();
  Timer? _captureTimer;
  Timer? _popupTimer;
  Timer? _countdownTimer;
  Timer? _cleanupTimer;
  late final CameraService cameraService;
  late final ChatWidget chatWidget = ChatWidget(key: GlobalKey<ChatWidgetState>());
  bool _showArrow = true;
  Timer? _blinkTimer;
  bool _showPopup = false;
  bool _isProcessingDetection = false;
  int _countdownSeconds = 5;
  bool _forceRerender = false;
  html.VideoElement? _videoElement;


  @override
  void initState() {
    super.initState();
    print("initState called");
    cameraService = Provider.of<CameraService>(context, listen: false);
    // chatWidget = ChatWidget();
    _initializeCamera();
    _startBlinkTimer();
  }

  Future<void> _initializeCamera() async {
    try {
      await cameraService.initializeCamera();
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to initialize camera: $e';
      });
      print('Camera initialization error: $e');
    }
  }

  void _toggleCamera() {
    if (cameraService.isCameraStopped) {
      _reinitializeCamera(); // Use the new reinitialize method
    } else {
      _stopCamera();
    }
  }

  void _stopCamera() {
    _captureTimer?.cancel();
    setState(() {
      cameraService.cameraAvailable = false;
      _recognitionResult = 'Camera stopped';
      cameraService.isCameraStopped = true;
      cameraService.pauseCapture();
    });
  }
  
  void _startBlinkTimer() {
    _blinkTimer = Timer.periodic(Duration(milliseconds: 500), (timer) {
      setState(() {
        _showArrow = !_showArrow;
      });
    });
  }

  // Add popup show/hide methods
  void _showPopupDialog() {
    if (!_showPopup && !_isProcessingDetection && cameraService.detectionResult.isNotEmpty) {
      setState(() {
        _showPopup = true;
        _isProcessingDetection = true;
        _countdownSeconds = 5;
      });

      // Pause capture while showing popup
      if (!cameraService.isCameraStopped) {
        cameraService.pauseCapture();
      }

      // Start countdown timer
      _countdownTimer?.cancel();
      _countdownTimer = Timer.periodic(Duration(seconds: 1), (timer) {
        if (mounted && _countdownSeconds > 0) {
          setState(() {
            _countdownSeconds--;
          });
        }
      });

      // Set popup timer
      _popupTimer?.cancel();
      _popupTimer = Timer(Duration(seconds: 5), () {
        if (mounted) {
          _hidePopup();
        }
      });
    }
  }


  // Modify hidePopup
  void _hidePopup([bool checkstop = false]) {
    if (mounted) {
      setState(() {
        _showPopup = false;
        _isProcessingDetection = false;
        _countdownSeconds = 5;
      });

      _countdownTimer?.cancel();

      // Clear detection result
      cameraService.detectionResult = '';
      
      // Don't change camera stopped state if it's already running
      // if (cameraService.isCameraStopped) {
      //   cameraService.isCameraStopped = false;
      // }

      if (!checkstop)
      {
        // cameraService.cameraAvailable = true;
        // Resume capture after a brief delay
        Future.delayed(Duration(seconds: 2), () {
          if (mounted && !cameraService.isCameraStopped) {
            cameraService.resumeCapture();
          }
        });
      }
      else{
        print("đã vào true rồi");
        _toggleCamera();
      }
    }
  }

  Future<void> _reinitializeCamera() async {
    try {
      await cameraService.initializeCamera();
      setState(() {
        cameraService.isCameraStopped = false;
        cameraService.cameraAvailable = true;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to reinitialize camera: $e';
      });
      print('Camera reinitialization error: $e');
    }
  }

  // Add method to handle recording start
  void _startRecording() {
    if (chatWidget.key != null) {
      final chatWidgetState = (chatWidget.key as GlobalKey<ChatWidgetState>).currentState;
      chatWidgetState?.startRecording(); // Changed from startListening to startRecording
    }
    _hidePopup(true);
  }

  Widget buildPopup() {
    if (!_showPopup || _forceRerender) return Container();

    return Positioned.fill(
      child: Container(
        key: ValueKey<bool>(_forceRerender), // Force rebuild on rerender
        color: Color.fromRGBO(0, 0, 0, 0.5),
        child: Center(
          child: Container(
            width: MediaQuery.of(context).size.width * 0.4,
            margin: EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              boxShadow: [
                BoxShadow(
                  color: Color.fromRGBO(0, 0, 0, 0.2),
                  blurRadius: 10,
                  offset: Offset(0, 5),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (!_forceRerender) ...[
                  Padding(
                    padding: EdgeInsets.all(20),
                    child: Image.asset(
                      'assets/logo_clb_2.png',
                      height: MediaQuery.of(context).size.height * 0.3,
                      fit: BoxFit.contain,
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 20),
                    child: Text(
                      cameraService.detectionResult,
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.all(10),
                    child: Text(
                      'Closing in $_countdownSeconds seconds',
                      style: TextStyle(
                        fontSize: 15,
                        color: Colors.grey,
                      ),
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.all(20),
                    child: ElevatedButton(
                      onPressed: () {
                        _startRecording();
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        padding: EdgeInsets.symmetric(
                          horizontal: 30,
                          vertical: 15,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(25),
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.mic, color: Colors.white),
                          SizedBox(width: 8),
                          Text(
                            'Start Recording',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 20,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _captureTimer?.cancel();
    _popupTimer?.cancel();
    _countdownTimer?.cancel();
    _cleanupTimer?.cancel();
    _blinkTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cameraService = Provider.of<CameraService>(context);
    final screenheight = MediaQuery.of(context).size.height;
    final screenwidth = MediaQuery.of(context).size.width;

    // Show popup when detection result changes
    if (cameraService.detectionResult.isNotEmpty && !_showPopup && !_isProcessingDetection) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _showPopupDialog();
      });
    }

    bool shouldShowResult = cameraService.cameraAvailable && 
      screenheight > 100 && 
      screenwidth > 100;

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        elevation: 20,
        backgroundColor: Colors.white, 
        titleSpacing: screenheight * 0.01,
        title: Row(
          children: [
            // Nhóm 1: Chiếm 4 phần
            Expanded(
              flex: 4,
              child: Container(
                margin: EdgeInsets.only(right: screenwidth * 0.005),
                height: screenheight * 0.08, // Đặt chiều cao cho Container
                decoration: BoxDecoration(
                  color: ui.Color.fromARGB(255, 171, 226, 255), // Màu nền
                  borderRadius: BorderRadius.circular(16.0), // Bo góc
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.5),
                      spreadRadius: 4,
                      blurRadius: 5,
                      offset: const Offset(0, 3),
                    ),
                  ],
                ),
                child: Stack(
                  alignment: Alignment.center, // Canh giữa toàn bộ nội dung
                  children: [
                    // Logo ở bên trái
                    Align(
                      alignment: Alignment.centerLeft,
                      child: Padding(
                        padding: EdgeInsets.only(left: screenwidth * 0.01),
                        child: Image.asset(
                          'assets/logo_clb_2.png',
                          height: screenheight * 0.06, // Giảm chiều cao logo
                          width: screenwidth * 0.12,
                        ),
                      ),
                    ),
                    // Chữ "Ohmni" nằm giữa
                    Center(
                      child: Text(
                        'Ohmni',
                        style: TextStyle(
                          fontFamily: 'new_font',
                          fontWeight: FontWeight.bold,
                          fontSize: screenwidth * 0.04,
                          shadows: const [
                            Shadow(
                              offset: Offset(2.0, 2.0),
                              blurRadius: 3.0,
                              color: Colors.grey,
                            ),
                          ],
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            // Nhóm 2: Chiếm 1 phần
            Expanded(
              flex: 1,
              child: Container(
                // margin: EdgeInsets.only(left: screenwidth * 0.01),
                height: screenheight * 0.08, // Đặt chiều cao cho Container
                decoration: BoxDecoration(
                  color: ui.Color.fromARGB(255, 171, 226, 255), // Màu nền
                  borderRadius: BorderRadius.circular(16.0), // Bo góc
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.5),
                      spreadRadius: 4,
                      blurRadius: 5,
                      offset: const Offset(0, 3),
                    ),
                  ],
                ),
                child: Center(
                  child: Text(
                    'Chat Bot',
                    style: TextStyle(
                      fontFamily: 'new_font',
                      fontWeight: FontWeight.bold,
                      fontSize: screenwidth * 0.03,
                      shadows: const [
                        Shadow(
                          offset: Offset(2.0, 2.0),
                          blurRadius: 3.0,
                          color: Colors.grey,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),

      body: Stack(
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                flex: 4,
                child: Container(
                  margin: EdgeInsets.all(screenwidth * 0.01),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Colors.grey.shade200,
                        const Color.fromARGB(255, 109, 190, 244),
                      ],
                    ),
                  ),
                  clipBehavior: Clip.hardEdge,
                  child: Stack(
                    children: [
                      _showPopup
                        ? buildPopup()
                        : Column(
                            children: [
                              Expanded(
                                    flex: 10,
                                    child: cameraService.cameraAvailable
                                        ? Container(
                                            decoration: BoxDecoration(
                                              borderRadius: BorderRadius.circular(16),
                                            ),
                                            child: const HtmlElementView(
                                              viewType: 'videoElement',
                                            ),
                                          )
                                        : Container(
                                            decoration: BoxDecoration(
                                              color: Colors.black,
                                              borderRadius: BorderRadius.circular(16),
                                              boxShadow: [
                                                BoxShadow(
                                                  color: Colors.grey.withOpacity(0.5),
                                                  spreadRadius: 4,
                                                  blurRadius: 5,
                                                  offset: const Offset(0, 3),
                                                ),
                                              ],
                                            ),
                                            child: Center(
                                              child: Column(
                                                mainAxisAlignment: MainAxisAlignment.center,
                                                children: [
                                                  const Text(
                                                    'Camera not available',
                                                    style: TextStyle(
                                                      color: Colors.white,
                                                      fontSize: 16,
                                                    ),
                                                  ),
                                                  if (_errorMessage.isNotEmpty)
                                                    Padding(
                                                      padding: const EdgeInsets.only(top: 8.0),
                                                      child: Text(
                                                        _errorMessage,
                                                        style: const TextStyle(
                                                          color: Colors.red,
                                                          fontSize: 12,
                                                        ),
                                                      ),
                                                    ),
                                                ],
                                              ),
                                            ),
                                          ),
                                  ),
                              Expanded(
                              flex: 1,
                              child: Center(
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    FloatingActionButton(
                                      onPressed: () {
                                        _toggleCamera();
                                        print("status camera: ${(cameraService.isCameraStopped)}");
                                      },
                                      child: Container(
                                        width: screenwidth * 0.5,
                                        height: screenwidth * 0.5,
                                        child: (cameraService.isCameraStopped
                                            ? Image.asset(
                                                'assets/off_live.png',
                                                height: screenwidth * 0.7,
                                                width: screenwidth * 0.7,
                                              )
                                            : Image.asset(
                                                'assets/live.png',
                                                height: screenwidth * 0.7,
                                                width: screenwidth * 0.7,
                                              )),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              ),
                            ]
                        ),
                    ],
                  ),
                ),
              ),
              Expanded(
                flex: 1,
                child: chatWidget,
              ),
            ],
          ),
        ],
      ),
    );
  }
}




