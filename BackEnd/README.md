# Backend

## Formateo numérico ATC

La función `format_atc_number` convierte un valor en su fraseo aeronáutico en español.

Reglas principales:
- **Pistas y QNH:** se pronuncian dígitos individuales. Ej.: `28` → "dos ocho", `3006` → "tres cero cero seis".
- **Frecuencias:** dígitos separados con la palabra "decimal". Ej.: `118.3` → "uno uno ocho decimal tres".
- **Altitudes:** miles y cientos en palabras. Ej.: `4700` → "cuatro mil setecientos".

Esta función se usa en `atc_phrase()` y en el endpoint `/tts` para asegurar que Polly pronuncie correctamente los números.
