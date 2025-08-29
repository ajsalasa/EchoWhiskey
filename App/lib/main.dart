import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:dio/dio.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';
import 'speech_service.dart';

void main() {
  runApp(const EcoWhiskyApp());
}

class EcoWhiskyApp extends StatelessWidget {
  const EcoWhiskyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EcoWhisky ATC',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF36D892)),
        useMaterial3: true,
      ),
      home: const TurnScreen(),
    );
  }
}

/// ===== Models (alineados a FastAPI) =====
class Contexto {
  final String airport;
  final String fase; // "superficie" | "torre" | "coco_app" | "coco_control" | "coco_radio"
  final String? runwayActual;
  final String? qnh;
  final int? vientoDir;
  final int? vientoVel;

  const Contexto({
    this.airport = 'MRPV',
    required this.fase,
    this.runwayActual,
    this.qnh,
    this.vientoDir,
    this.vientoVel,
  });

  Map<String, dynamic> toJson() => {
        'airport': airport,
        'fase': fase,
        'runway_actual': runwayActual,
        'qnh': qnh,
        'viento_dir': vientoDir,
        'viento_vel': vientoVel,
      };
}

class TurnInBody {
  final String textoAlumno;
  final Contexto contexto;
  const TurnInBody({required this.textoAlumno, required this.contexto});
  Map<String, dynamic> toJson() => {
        'texto_alumno': textoAlumno,
        'contexto': contexto.toJson(),
      };
}

class TurnOutBody {
  final String intent;
  final Map<String, dynamic> slots;
  final List<dynamic> missing;
  final String feedbackMicro;
  final String atc;
  final String fase;

  const TurnOutBody({
    required this.intent,
    required this.slots,
    required this.missing,
    required this.feedbackMicro,
    required this.atc,
    required this.fase,
  });

  factory TurnOutBody.fromJson(Map<String, dynamic> json) => TurnOutBody(
        intent: json['intent'] ?? '',
        slots: Map<String, dynamic>.from(json['slots'] ?? const {}),
        missing: List<dynamic>.from(json['missing'] ?? const []),
        feedbackMicro: json['feedback_micro'] ?? '',
        atc: json['atc'] ?? '',
        fase: json['fase'] ?? '',
      );
}

/// ===== Servicio HTTP =====
class TurnService {
  final Dio _dio;
  TurnService(this._dio);

  // Cambia con --dart-define si usas dispositivo físico:
  // flutter run -d ios --dart-define=BACKEND_BASE_URL=http://TU-IP-LAN:8000/
  static const String kDefaultBaseUrl =
      String.fromEnvironment('BACKEND_BASE_URL', defaultValue: 'http://127.0.0.1:8000/');

  factory TurnService.standard() {
    final dio = Dio(BaseOptions(
      baseUrl: kDefaultBaseUrl,
      connectTimeout: const Duration(seconds: 3),
      receiveTimeout: const Duration(seconds: 6),
      headers: {'Content-Type': 'application/json'},
    ));
    return TurnService(dio);
  }

  Future<TurnOutBody> turn(TurnInBody body) async {
    final resp = await _dio.post('/turn', data: body.toJson());
    return TurnOutBody.fromJson(resp.data as Map<String, dynamic>);
  }

  /// Pide audio neural (Polly) al backend /tts y devuelve bytes (WAV).
  Future<Uint8List> ttsNeural(String text,
      {String voiceId = 'Pedro', double rate = 0.88, int pitch = -6}) async {
    final resp = await _dio.post<List<int>>(
      '/tts',
      data: {'text': text, 'voice_id': voiceId, 'rate': rate, 'pitch': pitch},
      options: Options(responseType: ResponseType.bytes),
    );
    final data = resp.data ?? <int>[];
    if (data.isEmpty) {
      throw Exception('Respuesta vacía de /tts');
    }
    return Uint8List.fromList(data);
  }
}

/// ===== Pantalla principal =====
class TurnScreen extends StatefulWidget {
  const TurnScreen({super.key});
  @override
  State<TurnScreen> createState() => _TurnScreenState();
}

class _TurnScreenState extends State<TurnScreen> {
  late final TurnService service;
  final AudioPlayer player = AudioPlayer();
  late final SpeechService _speech;
  String _fase = 'torre';
  String _atc = '';
  String _feedback = '';
  int? _vientoDir = 80;
  int? _vientoVel = 12;
  String? _qnh = '3003';
  bool _isPlaying = false;
  String _recognized = '';

  @override
  void initState() {
    super.initState();
    service = TurnService.standard();
    player.onPlayerComplete.listen((_) => setState(() => _isPlaying = false));
    _speech = SpeechService();
    _speech.recognized.addListener(() {
      setState(() {
        _recognized = _speech.recognized.value;
      });
    });
    _speech.error.addListener(() {
      final err = _speech.error.value;
      if (err != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(err)));
      }
    });
  }

  /// Guarda bytes en archivo temporal y devuelve la ruta.
  Future<String> _writeTempAudio(Uint8List bytes, {String ext = 'wav'}) async {
    final dir = await getTemporaryDirectory();
    final path =
        '${dir.path}/ecowhisky_tts_${DateTime.now().microsecondsSinceEpoch}.$ext';
    final file = File(path);
    await file.writeAsBytes(bytes, flush: true);
    return path;
  }

  Future<void> _send(String texto) async {
    setState(() {
      _atc = '';
      _feedback = '';
    });
    try {
      final out = await service.turn(
        TurnInBody(
          textoAlumno: texto,
          contexto: Contexto(
            fase: _fase,
            qnh: _qnh,
            vientoDir: _vientoDir,
            vientoVel: _vientoVel,
          ),
        ),
      );
      setState(() {
        _atc = out.atc;
        _feedback = out.feedbackMicro;
      });

      if (_atc.isNotEmpty) {
        final bytes = await service.ttsNeural(_atc); // voz neural (wav)
        final path = await _writeTempAudio(bytes, ext: 'wav');
        await player.stop();
        setState(() => _isPlaying = true);
        await player.play(DeviceFileSource(path)); // ← iOS: desde archivo
      }
    } catch (e) {
      setState(() => _feedback = 'Error: $e');
    }
  }

  Future<void> _stopAndSend() async {
    await _speech.stop();
    final text = _recognized.trim();
    setState(() {
      _recognized = '';
    });
    if (text.isNotEmpty) {
      await _send(text);
    }
  }

  @override
  void dispose() {
    player.dispose();
    _speech.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('EcoWhisky ATC — iOS')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            GestureDetector(
              onTapDown: _isPlaying ? null : (_) => _speech.start(),
              onTapUp: _isPlaying ? null : (_) => _stopAndSend(),
              onTapCancel: _isPlaying ? null : _stopAndSend,
              child: Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primary,
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.mic, size: 60, color: Colors.white),
              ),
            ),
            const SizedBox(height: 24),
            Text('Feedback:', style: Theme.of(context).textTheme.titleMedium),
            Container(
              padding: const EdgeInsets.all(12),
              margin: const EdgeInsets.only(top: 8),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(_feedback),
            ),
          ],
        ),
      ),
    );
  }
}