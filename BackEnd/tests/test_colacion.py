import sys, types

# Stubs for external dependencies to allow importing main without heavy packages.
class _DummyApp:
    def get(self, *a, **k):
        def wrapper(f):
            return f
        return wrapper
    def post(self, *a, **k):
        def wrapper(f):
            return f
        return wrapper

sys.modules.setdefault("fastapi", types.SimpleNamespace(FastAPI=lambda *a, **k: _DummyApp(), Response=object))

class _BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

sys.modules.setdefault("pydantic", types.SimpleNamespace(BaseModel=_BaseModel))
sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *a, **k: types.SimpleNamespace()))
botocore_ex = types.SimpleNamespace(ClientError=Exception)
sys.modules.setdefault("botocore", types.SimpleNamespace(exceptions=botocore_ex))
sys.modules.setdefault("botocore.exceptions", botocore_ex)
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
numpy_stub = types.SimpleNamespace(
    ndarray=object,
    frombuffer=lambda *a, **k: None,
    float32=float,
    int16=int,
    ones=lambda *a, **k: None,
    sqrt=lambda *a, **k: None,
    maximum=lambda *a, **k: None,
    convolve=lambda *a, **k: None,
    log10=lambda *a, **k: None,
    random=types.SimpleNamespace(normal=lambda *a, **k: None),
    mean=lambda *a, **k: None,
    exp=lambda *a, **k: None,
    linspace=lambda *a, **k: None,
    concatenate=lambda *a, **k: None,
    clip=lambda *a, **k: None,
)
sys.modules.setdefault("numpy", numpy_stub)
scipy_signal = types.SimpleNamespace(butter=lambda *a, **k: None, sosfiltfilt=lambda *a, **k: None)
sys.modules.setdefault("scipy", types.SimpleNamespace(signal=scipy_signal))
sys.modules.setdefault("scipy.signal", scipy_signal)
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *a, **k: {}))

from .. import main


def test_required_slots_solicitar_rodaje():
    assert main.required_slots_for("superficie", "solicitar_rodaje") == ["indicativo", "pista"]


def test_turn_solicitar_rodaje_ok_sin_qnh(monkeypatch):
    monkeypatch.setattr(main, "llm_extract", lambda s, fase: {})
    ctx = main.Contexto(fase="superficie")
    texto = "tobias superficie alfa bravo charlie solicito rodaje a la pista 10"
    out = main.turn(main.TurnIn(texto_alumno=texto, contexto=ctx))
    assert out.missing == []
    assert out.slots["pista"] == "10"


def test_turn_solicitar_rodaje_falta_pista(monkeypatch):
    monkeypatch.setattr(main, "llm_extract", lambda s, fase: {})
    ctx = main.Contexto(fase="superficie")
    texto = "tobias superficie alfa bravo charlie solicito rodaje"
    out = main.turn(main.TurnIn(texto_alumno=texto, contexto=ctx))
    assert out.missing == ["pista"]
