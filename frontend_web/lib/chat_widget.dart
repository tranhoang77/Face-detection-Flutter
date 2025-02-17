import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'dart:html' as html;
import 'dart:async'; // Add this for Completer
import 'chat_message.dart';
import 'dart:typed_data'; // For Uint8List

class ChatWidget extends StatefulWidget {
  const ChatWidget({Key? key}) : super(key: key);

  @override
  ChatWidgetState createState() => ChatWidgetState();
}

class ChatWidgetState extends State<ChatWidget> with SingleTickerProviderStateMixin {
  List<ChatMessage> _messages = [];
  final _textController = TextEditingController();
  final _scrollController = ScrollController();
  bool _isLoading = false;
  bool _isRecording = false;
  html.MediaRecorder? _mediaRecorder;
  final List<html.Blob> _audioChunks = [];
  
  late AnimationController _animationController;
  
  @override
  void initState() {
    super.initState();
    _initAnimationController();
  }

  void _initAnimationController() {
    _animationController = AnimationController(
      duration: const Duration(seconds: 1),
      vsync: this,
    );
  }

    // Start recording audio
  Future<void> startRecording() async {
    try {
      final stream = await html.window.navigator.mediaDevices?.getUserMedia({'audio': true});
      if (stream != null) {
        _mediaRecorder = html.MediaRecorder(stream);

        // Capture audio chunks
        _mediaRecorder?.addEventListener('dataavailable', (event) {
          if (event is html.BlobEvent) {
            _audioChunks.add(event.data!);
          }
        });

        // Handle stop event
        _mediaRecorder?.addEventListener('stop', (_) async {
          if (_audioChunks.isNotEmpty) {
            final audioBlob = html.Blob(_audioChunks);
            _audioChunks.clear();
            final audioData = await _blobToUint8List(audioBlob);
            await _sendAudioToBackend(audioData);
          }
        });

        _mediaRecorder?.start();
        setState(() {
          _isRecording = true;
        });
      } else {
        print("Unable to access the microphone.");
      }
    } catch (e) {
      print('Error starting recording: $e');
    }
  }

  /// Converts a Blob to Uint8List
  Future<Uint8List> _blobToUint8List(html.Blob blob) {
    final completer = Completer<Uint8List>();
    final reader = html.FileReader();

    reader.readAsArrayBuffer(blob);
    reader.onLoadEnd.listen((_) {
      completer.complete(reader.result as Uint8List);
    });

    return completer.future;
  }

  void addSystemMessage(String message) {
    setState(() {
      _messages.add(ChatMessage(
        messageContent: message,
        isUser: false,
      ));
    });
    _scrollToBottom();
  }

  Future<void> stopRecording() async {
    if (!_isRecording) return;

    try {
      final recorder = _mediaRecorder;
      if (recorder != null && recorder.state == 'recording') {
        recorder.stop();
        _animationController.reset();
      }
    } catch (e) {
      print('Error stopping recording: $e');
    } finally {
      setState(() => _isRecording = false);
    }
  }

  // Send audio to speech-to-text backend
  Future<void> _sendAudioToBackend(Uint8List audioData) async {
    try {
      final uri = Uri.parse('https://aiclub.uit.edu.vn/gpu150/ohmni_be/api_chatbot');
      final request = http.MultipartRequest('POST', uri)
        ..headers['Content-Type'] = 'multipart/form-data'
        ..files.add(http.MultipartFile.fromBytes('file', audioData, filename: 'audio.webm'));

      final response = await request.send();
      print("has respond");
      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        print("gone to before decode");
        final jsonResponse = json.decode(responseBody);
        print("here 2");
        if (jsonResponse != null) {
          _messages.add(ChatMessage(messageContent: jsonResponse['message'], isUser: true));
          await _sendMessageToBackend(jsonResponse['message']);
        } else {
          addSystemMessage('Unable to process your request.');
        }
      } else {
        addSystemMessage('Error processing the audio: ${response.statusCode}');
      }
    } catch (e) {
      print('Error sending audio: $e');
      addSystemMessage('Error connecting to the server.');
    }
  }
  

  // // Send text message to chatbot backend
  // Future<void> _sendMessageToBackend(String message) async {
  //   try {
  //     setState(() => _isLoading = true);
      
  //     final response = await http
  //         .post(
  //           Uri.parse('https://aiclub.uit.edu.vn/gpu150/chatbot_uit_be/query'),
  //           headers: {'Content-Type': 'application/json; charset=UTF-8'},
  //           body: json.encode({'content': message}),
  //         )
  //         .timeout(const Duration(seconds: 30));

  //     if (response.statusCode == 200) {
  //       final Map<String, dynamic> responseData = json.decode(utf8.decode(response.bodyBytes));
  //       if (responseData.containsKey("result")) {
  //         addSystemMessage(responseData["result"]);
  //       } else {
  //         addSystemMessage("Sorry, I couldn't process your request properly.");
  //       }
  //     } else {
  //       addSystemMessage("Sorry, there was an error processing your request.");
  //     }
  //   } catch (e) {
  //     addSystemMessage("Sorry, there was an error connecting to the server.");
  //     print('Error occurred: $e');
  //   } finally {
  //     setState(() => _isLoading = false);
  //   }
  // }

  // API call function (unchanged)
  Future<void> _sendMessageToBackend(String prompt) async {
    final apiKey = 'AIzaSyCCRFZGeqj5IAGpw9OejUJ3ItkAER9JZnw';
    final url = Uri.parse(
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$apiKey');

    try {
      setState(() => _isLoading = true);
      
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          "contents": [
            {
              "parts": [
                {"text": prompt}
              ]
            }
          ]
        }),
      );

      if (response.statusCode == 200) {
        final responseData = json.decode(utf8.decode(response.bodyBytes));
        final text = responseData['candidates'][0]['content']['parts'][0]['text'];
        addSystemMessage(text);
      } else {
        addSystemMessage("Sorry, there was an error processing your request.");
        print('Failed with status: ${response.statusCode}');
        print('Body: ${response.body}');
      }
    } catch (e) {
      addSystemMessage("Sorry, there was an error connecting to the server.");
      print('Error occurred: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }


  void _sendMessage() async {
    final text = _textController.text.trim();
    if (text.isNotEmpty) {
      setState(() {
        _messages.add(ChatMessage(messageContent: text, isUser: true));
        _textController.clear();
      });
      _scrollToBottom();
      await _sendMessageToBackend(text);
    }
  }

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    }
  }

  void _deleteAllMessages() {
    setState(() {
      _messages.clear();
    });
  }

  // Recording indicator widget
  Widget _buildRecordingIndicator() {
    return AnimatedBuilder(
      animation: _animationController,
      builder: (context, child) {
        return Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.red.withOpacity(_animationController.value),
          ),
        );
      },
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final screenWidth = constraints.maxWidth;
        return Container(
          width: screenWidth,
          child: Container(
            margin: EdgeInsets.only(
              right: screenWidth * 0.05,
              top: screenWidth * 0.05,
              bottom: screenWidth * 0.05,
            ),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              gradient: const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Color.fromARGB(255, 252, 252, 252),
                  Color.fromARGB(255, 109, 190, 244),
                ],
              ),
            ),
            child: Column(
              children: [
                if (_isRecording)
                  Container(
                    padding: EdgeInsets.symmetric(vertical: 8),
                    color: Colors.black.withOpacity(0.1),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        _buildRecordingIndicator(),
                        SizedBox(width: 8),
                        Text(
                          'Đang ghi âm...',
                          style: TextStyle(
                            color: Colors.red,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                Expanded(
                  child: ListView.builder(
                    controller: _scrollController,
                    padding: EdgeInsets.all(16.0),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      final message = _messages[index];
                      return Align(
                        alignment: message.isUser
                            ? Alignment.centerRight
                            : Alignment.centerLeft,
                        child: Container(
                          constraints: BoxConstraints(
                            maxWidth: screenWidth * 0.7,
                          ),
                          margin: EdgeInsets.only(
                            bottom: 8.0,
                            left: message.isUser ? screenWidth * 0.15 : 0,
                            right: message.isUser ? 0 : screenWidth * 0.15,
                          ),
                          decoration: BoxDecoration(
                            color: message.isUser
                                ? Colors.blue[200]
                                : Colors.grey[200],
                            borderRadius: BorderRadius.circular(16.0),
                          ),
                          padding: const EdgeInsets.all(12.0),
                          child: Text(
                            message.messageContent,
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      );
                    },
                  ),
                ),
                if (_isLoading)
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: CircularProgressIndicator(),
                  ),
                Padding(
                  padding: EdgeInsets.all(screenWidth * 0.02),
                  child: Row(
                    children: [
                      Container(
                        width: screenWidth * 0.12,
                        height: screenWidth * 0.12,
                        child: ElevatedButton(
                          onPressed: _isRecording ? stopRecording : startRecording,
                          style: ElevatedButton.styleFrom(
                            shape: CircleBorder(),
                            padding: EdgeInsets.zero,
                            backgroundColor: _isRecording 
                              ? Colors.red 
                              : const Color.fromARGB(255, 35, 41, 231),
                          ),
                          child: Icon(
                            _isRecording ? Icons.mic : Icons.mic_none,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      SizedBox(width: screenWidth * 0.02),
                      Expanded(
                        child: TextField(
                          controller: _textController,
                          decoration: InputDecoration(
                            fillColor: Colors.white,
                            filled: true,
                            hintText: 'Nhập tin nhắn...',
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(25.0),
                            ),
                          ),
                          onSubmitted: (_) => _sendMessage(),
                        ),
                      ),
                      SizedBox(width: screenWidth * 0.02),
                      Container(
                        width: screenWidth * 0.12,
                        height: screenWidth * 0.12,
                        child: ElevatedButton(
                          onPressed: _isLoading ? null : _sendMessage,
                          style: ElevatedButton.styleFrom(
                            shape: CircleBorder(),
                            padding: EdgeInsets.zero,
                            backgroundColor: const Color.fromARGB(255, 35, 41, 231),
                          ),
                          child: Icon(Icons.send, color: Colors.white),
                        ),
                      ),
                      SizedBox(width: screenWidth * 0.02),
                      Container(
                        width: screenWidth * 0.12,
                        height: screenWidth * 0.12,
                        child: ElevatedButton(
                          onPressed: _deleteAllMessages,
                          style: ElevatedButton.styleFrom(
                            shape: CircleBorder(),
                            padding: EdgeInsets.zero,
                            backgroundColor: const Color.fromARGB(255, 50, 27, 228),
                          ),
                          child: Icon(Icons.delete, color: Colors.white),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}