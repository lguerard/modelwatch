# docker:cli = Docker client + compose plugin (modelwatch runs `docker exec`
# for exec suites and probes, and `docker compose up` to adopt a model), on Alpine.
FROM docker:29-cli

# tzdata: without it TZ is ignored and the schedule would be read as UTC.
RUN apk add --no-cache python3 tzdata

ARG VERSION=dev
LABEL org.opencontainers.image.title="modelwatch" \
      org.opencontainers.image.description="Is there a better local model for your apps? Scored on your own tasks, one tap to switch" \
      org.opencontainers.image.source="https://github.com/lguerard/modelwatch" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      updock.enable="false"

WORKDIR /app
COPY modelwatch.py ui.html ./
COPY plugins ./plugins
COPY fixtures ./fixtures
COPY examples ./examples
ENV MODELWATCH_STATE_FILE=/data/state.json MODELWATCH_VERSION=${VERSION} PYTHONUNBUFFERED=1
# The dependency-free tests run at build time: a broken image is never produced.
RUN python3 -I modelwatch.py --selftest
VOLUME /data
EXPOSE 8097
HEALTHCHECK --interval=1m --timeout=5s CMD wget -q -O /dev/null http://127.0.0.1:8097/health || exit 1

CMD ["python3", "/app/modelwatch.py"]
