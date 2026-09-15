# Python 3.12 : scikit-learn 1.9 et numpy 2.4 exigent >= 3.11.
# La version doit permettre d'installer exactement les versions listees dans
# requirements.txt, sans quoi model.pkl ne serait pas deserialisable.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py model.pkl feature_info.pkl ./

# Aucun port n'est code en dur. Render injecte PORT (10000 par defaut) et cette
# valeur prend le pas sur l'ENV ci-dessous ; en local et sous Docker classique,
# le service reste servi sur 8000.
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen('http://localhost:'+p+'/health').status==200 else 1)"

# Forme shell volontaire (et non exec) : c'est le shell qui substitue ${PORT:-8000}
# au demarrage. En forme exec, la variable serait passee telle quelle a uvicorn.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
