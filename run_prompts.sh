#!/bin/bash

TRANSCRIPTS="НД транскрипты"
LOG_FILE="model_responses_whisper.log"
PYTHON_SCRIPT="model_prompt_simple.py"

touch "$LOG_FILE"

for FILE in "$TRANSCRIPTS"/*.txt; do
        if [[ -f "$FILE" ]]; then
                echo "Processing: $FILE"

                TEXT=$(cat "$FILE")

#                 PROMPT=$(cat <<EOF
# Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи.
# Запиши протокол встречи на основе представленного транскрипта по следующему формату:
# 1. 10 ключевых тезисов встречи
# 2. Принятые решения, ответственные за их исполнения, сроки
# 3. Ближайшие шаги. Отметь наиболее срочные задачи. Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.

# $TEXT
# EOF
# )


                START_TIME=$(date +%s)

                RESPONSE=$(python3 "$PYTHON_SCRIPT" "$TRANSCRIPTS/$FILE")

                END_TIME=$(date +%s)
                ELAPSED_TIME=$((END_TIME - START_TIME))

                echo "File: $FILE" >> "$LOG_FILE"
                echo "Response: $RESPONSE" >> "$LOG_FILE"
                echo "Time Taken: ${ELAPSED_TIME}s" >> "$LOG_FILE"
                echo "-----------------------------------------" >> "$LOG_FILE"

                echo "Completed: $FILE (Time Taken: ${ELAPSED_TIME}s)"
        fi
done

echo "Transcripts processed. Results saved in $LOG_FILE"

