# Pipeline Engavetados (um caso por dia, 12h BRT)

Conta Instagram: casosengavetados (Windsor, connector `instagram`, account `17841480368237689`).
Este repositório é público: o Instagram baixa o MP4 por `raw.githubusercontent.com/laurajube7/videos-casosengavetados/main/<pasta>/<arquivo>`.

## Regras editoriais (acordadas com a dona do canal)
- Casos: Brasil e mundo; resolvidos, arquivados e sem solução.
- Nunca nomear suspeito sem condenação ("investigado"/"acusado"). Violência sexual contra menores: NUNCA. Contra adultos: nunca automático, sem detalhe gráfico, sem identificar vítima.
- Cada fato precisa de >= 2 fontes. Um segundo agente independente revisa fatos antes de publicar.
- Legenda + hashtags; fontes em comentário (a API não fixa: avisar a dona); pergunta final para gerar comentários.
- Primeiro quadro do vídeo = capa (pasta amarela, CASO #n, nome, país·ano, carimbo). Subir também capa.jpg e usar `cover_url`.
- Publicar só com autorização da dona; a regra vigente: sem resposta dela até o horário, a decisão é do agente (revisar e publicar se tudo estiver certo; senão pular o dia e avisar).

## Passos (ver modelo em pipeline/modelo_caso01)
1. Escolher caso, pesquisar (>=2 fontes), escrever roteiro (~14-16 frases, ~75-85 s) com tags de tom do ElevenLabs ([whispering], [pause], [firm]...).
2. Voz: ElevenLabs, voz Will (voice_id CstacWqMhJQlnfLPxRG4), model eleven_v4, generations_count 1 (gasta créditos, nunca repetir). O áudio só sai pelo navegador do PC da dona (o ambiente na nuvem não alcança elevenlabsusercontent.com): pegar a `content_url` com creative_get_flow_run_status, abrir no navegador embutido e salvar em ~/Downloads (fetch + <a download>), depois device_stage_files.
3. `align.py`: detecta pausas (silencedetect) e alinha o texto ao áudio (programação dinâmica) -> timing.json. Ajustar a lista SENT.
4. Animação: copiar modelo_caso01/index.html e reescrever as cenas (função scene(...)) para o caso novo, sincronizando pop() com wt(frase, palavra). Capa no #cover. Conferir quadros com qa.py (seek) em vários tempos.
5. `rec.py` grava em tempo real (~92 s) com marcador magenta; detectar o marcador, cortar -ss (marcador+0.18) e juntar áudio com atraso 0.42 s.
6. `sound.py`: música sintetizada + efeitos sincronizados; loudnorm -14 LUFS.
7. Subir MP4+capa para uma pasta casoNN/ neste repo; Windsor create_video_post (video_url, cover_url, caption); create_comment com as fontes.
