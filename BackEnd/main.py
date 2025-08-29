# main.py
# ATC Trainer (MRPV, ES) — FastAPI + LLM (Bedrock) + Polly Neural + Radio-FX
# Endpoints:
#   POST /turn  -> Analiza la frase (LLM), detecta errores, extrae slots y responde con frase ATC determinista
#   POST /tts   -> TTS (Polly Neural) + Radio-FX (WAV 16 kHz)
#   GET  /health
#
# Ejecutar:
#   pip install fastapi uvicorn pydantic pyyaml boto3 python-dotenv numpy scipy
#   uvicorn main:app --reload

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal,Union
import re, yaml, os, io, json, random
import numpy as np
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from scipy.signal import butter, sosfiltfilt
from html import escape

# ---------- Config ----------
load_dotenv()
AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
POLLY_VOICE       = os.getenv("POLLY_VOICE_ID", "Mia")  # es-MX
BEDROCK_MODEL_ID  = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

polly    = boto3.client("polly", region_name=AWS_REGION)
bedrock  = boto3.client("bedrock-runtime", region_name=AWS_REGION)

app = FastAPI(title="EcoWhisky ATC — Backend ES (MRPV)")

# ---------- Datos: aeropuerto ----------
DEFAULT_AIRPORT_YAML = """
icao: MRPV
nombre: "Tobías Bolaños (Pavas)"
zona_horaria: America/Costa_Rica
elevacion_ft: 3287
pistas:
  - id: "10/28"
    longitud_m: 1566
patron:
  altitud_ft: 4700
comunicaciones:
  emergencia: "121.5"
  superficie: "121.7"
  torre: "118.3"
  coco_aproximacion: "120.5"
  coco_control: "119.6"
  coco_radio: "126.8"
"""
def load_airport(icao: str = "MRPV") -> Dict:
    path = os.path.join("data", f"{icao}.es.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return yaml.safe_load(DEFAULT_AIRPORT_YAML)
AIRPORT = load_airport("MRPV")

# ---------- Plantillas ATC ----------
TEMPLATES = {
    "superficie": {
        "falta_dato": "¿{faltante}? {indicativo} confirme.",
        "rodaje": "{indicativo} recibido, pista en uso {pista}, ruede {ruta} al punto de espera pista {pista}, QNH {qnh}, mantenga posición corta.",
        "ack": "{indicativo} recibido."
    },
    "torre": {
        "falta_dato": "¿{faltante}? {indicativo} confirme.",
        "line_up_wait": "{indicativo}, ruede a posición y mantenga pista {pista}.",
        "despegue_autorizado": "{indicativo}, viento {viento_dir} grados {viento_vel} nudos, autorizado para despegar pista {pista}, mantenga {instruccion_post}.",
        "transferir_coco_app": "{indicativo} recibido, contacte COCO Aproximación {frecuencia}."
    },
    "coco_app": {
        "ack": "{indicativo} COCO Aproximación, contacto radar, continúe su ascenso para {nivel_ft} pies, QNH {qnh}, llame ingresando a la zona {zona}."
    },
    "coco_radio": {
        "ack": "{indicativo} recibido, QNE dos nueve nueve dos, sin tránsito notificado, notifiquen {minutos} minutos de vuelo normal."
    }
}

# ---------- Normalización & extracción básica ----------
NATO = {
    "alfa":"A","alpha":"A","noviembre":"N","india":"I","bravo":"B","charlie":"C","delta":"D",
    "echo":"E","eco":"E","foxtrot":"F","golf":"G","hotel":"H","juliet":"J","kilo":"K",
    "lima":"L","mike":"M","oscar":"O","papa":"P","quebec":"Q","romeo":"R","sierra":"S",
    "tango":"T","uniform":"U","victor":"V","whiskey":"W","whisky":"W","xray":"X","x-ray":"X",
    "yankee":"Y","zulu":"Z"
}
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("cocoaximación", "coco aproximación").replace("paz", "pavas")
    s = s.replace("censo", "ascenso").replace("argo", "largo")
    s = re.sub(r"\s+", " ", s)
    return s
def extract_callsign(s: str) -> Optional[str]:
    words = s.split()
    letters = []
    for w in words:
        if w in NATO: letters.append(NATO[w])
        elif re.fullmatch(r"[a-zA-Z]{3,5}", w): return w.upper()
    if 2 <= len(letters) <= 6: return "".join(letters)
    return None
def extract_runway(s: str) -> Optional[str]:
    if "pista 10" in s or "pista uno cero" in s or re.search(r"\b10\b", s): return "10"
    if "pista 28" in s or "pista veintiocho" in s or re.search(r"\b28\b", s): return "28"
    return None
def extract_altitude_ft(s: str) -> Optional[int]:
    m = re.search(r"(\d{3,5})\s*pies", s)
    if m: return int(m.group(1))
    if "siete mil" in s: return 7000
    if "cinco mil" in s: return 5000
    if "cuatro mil" in s: return 4000
    return None
def extract_zone(s: str) -> Optional[str]:
    if "zona eco" in s or "zona echo" in s: return "ECO"
    if "zona whisky" in s or "zona whiskey" in s: return "WHISKY"
    return None
def extract_qnh(s: str) -> Optional[str]:
    m = re.search(r"qnh\s*(\d{3,4})", s)
    if m: return f"{m.group(1)}"
    if "tres cero cero tres" in s: return "3003"
    if "tres cero cero cinco" in s: return "3005"
    if "tres cero cero seis" in s: return "3006"
    return None

# ---------- Modelos ----------
class Contexto(BaseModel):
    airport: str = "MRPV"
    fase: Literal["superficie", "torre", "coco_app", "coco_control", "coco_radio"]
    runway_actual: Optional[str] = None
    qnh: Optional[str] = None
    viento_dir: Optional[int] = None
    viento_vel: Optional[int] = None

class TurnIn(BaseModel):
    texto_alumno: str
    contexto: Contexto

class TurnOut(BaseModel):
    intent: str
    slots: Dict[str, Optional[str]]
    missing: List[str]
    feedback_micro: str
    atc: str
    fase: str
    env: Dict[str, Optional[str]]  # añadido: qnh/viento usados

class EnvOut(BaseModel):
    qnh: Optional[str] = None
    viento_dir: Optional[int] = None
    viento_vel: Optional[int] = None
    pista: Optional[str] = None

class TurnOut(BaseModel):
    intent: str
    slots: Dict[str, Optional[Union[str, int]]]   # 👈 admite nivel_ft:int
    missing: List[str]
    feedback_micro: str
    atc: str
    fase: str
    env: EnvOut                                   # 👈 tipa env correctamente

# ---------- Intents base (fallback) ----------
INTENTS = {
    "superficie": [
        ("abrir_plan", r"(abriendo|abrir).*plan de vuelo"),
        ("solicitar_rodaje", r"(solicito|solicitamos).*rodaje|rueda? al punto|taxi")
    ],
    "torre": [
        ("listo_despegue", r"(listo|listos).*(salida|despegue)"),
        ("reportar_altitud", r"(alcanz(a|ando)|pasando).*(pies)"),
    ],
    "coco_app": [
        ("ingresando_zona", r"(ingresando|entrando).*zona"),
    ],
    "coco_radio": [
        ("reporte_periodico", r"(tr[aá]nsito|vertical|al norte|al sur|dejando).*"),
    ]
}
def detect_intent_rule(fase: str, s: str) -> str:
    for intent, pat in INTENTS.get(fase, []):
        if re.search(pat, s): return intent
    return "ack"

# ---------- Entorno aleatorio (QNH/viento/pista) ----------
def gen_qnh_str() -> str:
    # 29.70–30.30 inHg -> 2970–3030
    val = random.randint(2970, 3030)
    return f"{val:04d}"
def gen_wind() -> Dict[str, int]:
    # Dirección 010–360, velocidad 3–18 kt (distribución simple)
    d = random.randint(5, 360)
    v = random.randint(3, 18)
    return {"dir": d, "vel": v}
def choose_runway_from_wind(viento_dir: Optional[int]) -> str:
    if viento_dir is None: return "10"
    return "10" if 40 <= viento_dir <= 219 else "28"
def ensure_env(ctx: Contexto) -> Contexto:
    if ctx.qnh is None: ctx.qnh = gen_qnh_str()
    if ctx.viento_dir is None or ctx.viento_vel is None:
        w = gen_wind()
        ctx.viento_dir = w["dir"]; ctx.viento_vel = w["vel"]
    if ctx.runway_actual is None:
        ctx.runway_actual = choose_runway_from_wind(ctx.viento_dir)
    return ctx

# ---------- LLM (Bedrock) ----------
LLM_SYSTEM = (
    "Eres instructor ATC en español para MRPV (Costa Rica). "
    "Analiza la transmisión libre del alumno y devuelve JSON ESTRICTO con:\n"
    '{ "intent": "<string>", '
    '"slots": {"indicativo": "<str|null>", "pista": "<10|28|null>", "nivel_ft": <int|null>, '
    '"zona": "<ECO|WHISKY|null>", "qnh": "<str|null>"}, '
    '"errores": ["<string>", "..."], '
    '"feedback_micro": "<string>" } '
    "No agregues texto fuera del JSON. No inventes QNH/ viento si el alumno no lo dijo."
)

def llm_extract(s: str, fase: str) -> Optional[Dict]:
    """
    Usa Amazon Nova Micro (Converse API) para extraer intent/slots/errores.
    Retorna {} si algo falla (el pipeline seguirá con reglas).
    """
    try:
        resp = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": LLM_SYSTEM + f" FASE={fase}."}],
            messages=[{"role": "user", "content": [{"text": s}]}],
            inferenceConfig={"maxTokens": 400, "temperature": 0.0, "topP": 0.9},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        # JSON estricto esperado; si hubiese espacios/ruido, intentamos aislar {...}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}
    except Exception as e:
        print(f"LLM error en Bedrock: {e}")
        return {}
# ---------- Frase ATC ----------
def atc_phrase(ctx: Contexto, slots: Dict[str, Optional[str]], intent: str, missing: List[str]) -> str:
    indicativo = slots.get("indicativo") or "Tráfico"
    fase = ctx.fase
    qnh = ctx.qnh or "3003"
    viento_dir = ctx.viento_dir or 80
    viento_vel = ctx.viento_vel or 12
    pista = ctx.runway_actual or slots.get("pista") or choose_runway_from_wind(ctx.viento_dir)

    if missing:
        return TEMPLATES[fase]["falta_dato"].format(faltante=" / ".join(missing), indicativo=indicativo)

    if fase == "superficie":
        if intent in ("abrir_plan", "solicitar_rodaje"):
            return TEMPLATES["superficie"]["rodaje"].format(indicativo=indicativo, pista=pista, ruta="A2-A", qnh=qnh)
        return TEMPLATES["superficie"]["ack"].format(indicativo=indicativo)

    if fase == "torre":
        if intent == "listo_despegue":
            instr = "rumbo de pista y notifique alcanzando cinco mil pies"
            return TEMPLATES["torre"]["despegue_autorizado"].format(
                indicativo=indicativo, viento_dir=viento_dir, viento_vel=viento_vel,
                pista=pista, instruccion_post=instr
            )
        if intent == "reportar_altitud":
            return TEMPLATES["torre"]["transferir_coco_app"].format(
                indicativo=indicativo, frecuencia=AIRPORT["comunicaciones"]["coco_aproximacion"]
            )
        return TEMPLATES["torre"]["line_up_wait"].format(indicativo=indicativo, pista=pista)

    if fase == "coco_app":
        nivel = slots.get("nivel_ft") or 7000
        zona = slots.get("zona") or "ECO"
        return TEMPLATES["coco_app"]["ack"].format(indicativo=indicativo, nivel_ft=nivel, qnh=qnh, zona=zona)

    if fase == "coco_radio":
        return TEMPLATES["coco_radio"]["ack"].format(indicativo=indicativo, minutos=30)

    return f"{indicativo} recibido."

def required_slots_for(fase: str, intent: str) -> List[str]:
    if fase == "superficie" and intent in ("abrir_plan", "solicitar_rodaje"): return ["indicativo", "pista", "qnh"]
    if fase == "torre" and intent == "listo_despegue": return ["indicativo", "pista"]
    if fase == "coco_app" and intent == "ingresando_zona": return ["indicativo", "zona", "nivel_ft"]
    return ["indicativo"]

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "airport": AIRPORT.get("icao", "MRPV")}

@app.post("/turn", response_model=TurnOut)
def turn(in_: TurnIn):
    # 1) Normaliza texto y asegura entorno aleatorio
    s = normalize_text(in_.texto_alumno)
    ctx = ensure_env(in_.contexto)

    # 2) Extracción por reglas (baseline)
    slots_rule = {
        "indicativo": extract_callsign(s),
        "pista": extract_runway(s),
        "nivel_ft": extract_altitude_ft(s),
        "zona": extract_zone(s),
        "qnh": extract_qnh(s),
    }
    intent_rule = detect_intent_rule(ctx.fase, s)

    # 3) LLM para extracción flexible + errores
    llm = llm_extract(s, ctx.fase) or {}
    intent = llm.get("intent") or intent_rule
    slots_llm = (llm.get("slots") or {})
    # merge: prioriza LLM si aporta valor, sino usa regla
    slots = {
        "indicativo": slots_llm.get("indicativo") or slots_rule["indicativo"],
        "pista": slots_llm.get("pista") or slots_rule["pista"],
        "nivel_ft": slots_llm.get("nivel_ft") or slots_rule["nivel_ft"],
        "zona": slots_llm.get("zona") or slots_rule["zona"],
        "qnh": slots_llm.get("qnh") or slots_rule["qnh"],
    }

    # 4) Validación de slots requeridos y feedback
    required = required_slots_for(ctx.fase, intent)
    missing = [r for r in required if not slots.get(r)]
    errores = llm.get("errores") or []
    fb_llm = llm.get("feedback_micro") or ""
    fb_rule = "Correcto." if not missing else f"Falta {', '.join(missing)} en su colación."
    feedback = (fb_llm.strip() or fb_rule) + (("" if not errores else " | " + "; ".join(errores)))

    # 5) Frase ATC determinista (con entorno generado)
    atc = atc_phrase(ctx, slots, intent, missing)

    env_used = EnvOut(
        qnh=ctx.qnh,
        viento_dir=ctx.viento_dir,
        viento_vel=ctx.viento_vel,
        pista=ctx.runway_actual,
    )
    return TurnOut(
        intent=intent,
        slots=slots,
        missing=missing,
        feedback_micro=feedback,
        atc=atc,
        fase=ctx.fase,
        env=env_used,            # 👈 ahora es EnvOut, no Dict[str,str]
    )

# ---------- TTS NEURAL + Radio-FX ----------
class TtsIn(BaseModel):
    text: str
    voice_id: Optional[str] = None
    rate: Optional[float] = 0.90   # 0.5–1.5 -> 60–140%
    pitch: Optional[int] = 0       # variación en %, 0 = sin cambio

def build_polly_ssml(text: str, rate: float = 0.90, pitch: int = 0) -> str:
    """Construye SSML para Polly Neural.

    Solo se modifica *rate* y opcionalmente *pitch*. El parámetro de tono se
    expresa como porcentaje (+/-) para evitar valores inválidos.
    """
    rate_pct = max(60, min(140, int(round(rate * 100))))
    pitch_val = max(-50, min(50, int(pitch)))  # limita variaciones extremas
    if rate_pct == 100 and pitch_val == 0:
        return f'<speak>{escape(text)}</speak>'
    attrs = []
    if rate_pct != 100:
        attrs.append(f'rate="{rate_pct}%"')
    if pitch_val != 0:
        attrs.append(f'pitch="{pitch_val}%"')
    attr_str = " ".join(attrs)
    return f'<speak><prosody {attr_str}>{escape(text)}</prosody></speak>'

def synthesize_pcm16_neural(text: str, voice_id: str, rate: float, pitch: int = 0) -> np.ndarray:
    ssml = build_polly_ssml(text, rate=rate, pitch=pitch)
    resp = polly.synthesize_speech(
        TextType="ssml", Text=ssml, VoiceId=voice_id,
        Engine="neural", OutputFormat="pcm", SampleRate="16000",
    )
    pcm_bytes = resp["AudioStream"].read()
    if not pcm_bytes: raise RuntimeError("Polly devolvió audio vacío (neural).")
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

def bp_filter(x: np.ndarray, sr=16000, lo=300, hi=3000, order=4) -> np.ndarray:
    sos = butter(order, [lo/(sr/2), hi/(sr/2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)
def comp_soft(x: np.ndarray, sr=16000, thresh_db=-14.0, ratio=3.0, win_ms=10) -> np.ndarray:
    n = max(1, int(sr * win_ms / 1000))
    kernel = np.ones(n, dtype=np.float32) / n
    rms = np.sqrt(np.maximum(1e-9, np.convolve(x**2, kernel, mode="same")))
    level_db = 20*np.log10(rms + 1e-9)
    over = np.maximum(0.0, level_db - thresh_db)
    gain_db = -over * (1.0 - 1.0/ratio)
    gain = 10**(gain_db/20.0)
    return x * gain
def add_hiss(x: np.ndarray, noise_db=-32.0) -> np.ndarray:
    rms_target = 10**(noise_db/20.0)
    n = np.random.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    n *= (rms_target / (np.sqrt(np.mean(n**2) + 1e-9)))
    return x + n
def squelch_tail(sr=16000, ms=70, noise_db=-28.0) -> np.ndarray:
    n = int(sr * ms / 1000)
    tail = np.random.normal(0.0, 1.0, size=n).astype(np.float32)
    tail *= (10**(noise_db/20.0)) / (np.sqrt(np.mean(tail**2) + 1e-9))
    env = np.exp(-np.linspace(0, 5, n)).astype(np.float32)
    return tail * env
def to_wav_bytes(x: np.ndarray, sr=16000) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    y = (x * 32767.0).astype(np.int16)
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y.tobytes())
    return buf.getvalue()

@app.post("/tts")
def tts_fx(in_: TtsIn):
    """Genera audio WAV (16 kHz) con efecto de radio.

    La salida se entrega como ``audio/wav``; para MP3 se requeriría codificación
    adicional.
    """
    voice = in_.voice_id or POLLY_VOICE
    try:
        x = synthesize_pcm16_neural(in_.text, voice, in_.rate or 0.9, in_.pitch or 0)
    except ClientError as e:
        return Response(status_code=502, content=str(e).encode("utf-8"), media_type="text/plain")
    except Exception as e:
        return Response(status_code=502, content=f"Polly neural error: {e}".encode("utf-8"), media_type="text/plain")
    x = bp_filter(x); x = comp_soft(x); x = add_hiss(x); x = np.concatenate([x, squelch_tail()], axis=0)
    wav = to_wav_bytes(x)
    return Response(content=wav, media_type="audio/wav")

@app.get("/health")
def health():
    return {"ok": True, "airport": AIRPORT.get("icao", "MRPV")}