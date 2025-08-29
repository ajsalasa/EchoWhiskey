import 'package:flutter/foundation.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

/// Service to handle speech recognition using [speech_to_text].
class SpeechService {
  final stt.SpeechToText _speech = stt.SpeechToText();

  /// Holds the last recognized words.
  final ValueNotifier<String> recognized = ValueNotifier('');

  /// Holds the last error message, if any.
  final ValueNotifier<String?> error = ValueNotifier(null);

  /// Starts listening for speech and updates [recognized].
  Future<void> start() async {
    try {
      final available = await _speech.initialize(
        onError: (e) => error.value = e.errorMsg,
      );
      if (!available) {
        error.value = 'Permiso de micrófono denegado o no disponible.';
        return;
      }
      await _speech.listen(onResult: (r) => recognized.value = r.recognizedWords);
    } catch (e) {
      error.value = '$e';
    }
  }

  /// Stops the speech recognition.
  Future<void> stop() async {
    try {
      await _speech.stop();
    } catch (e) {
      error.value = '$e';
    }
  }

  /// Releases resources.
  void dispose() {
    recognized.dispose();
    error.dispose();
    _speech.stop();
  }
}
