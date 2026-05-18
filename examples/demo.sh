#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pretty_print() {
  if command -v jq &>/dev/null; then
    echo "$1" | jq .
  else
    echo "$1"
  fi
}

echo "=== Document Q&A Demo ==="
echo "Target: $BASE_URL"
echo

# Health check
echo "--- Health Check ---"
HEALTH=$(curl -s "$BASE_URL/health")
pretty_print "$HEALTH"
echo

# Ingest document
echo "--- Ingesting Document ---"
DOC_TEXT=$(cat "$SCRIPT_DIR/sample_document.txt")
INGEST_RESPONSE=$(curl -s -X POST "$BASE_URL/documents" \
  -H "Content-Type: application/json" \
  -d "{\"document_text\": $(echo "$DOC_TEXT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}")

pretty_print "$INGEST_RESPONSE"
echo

DOCUMENT_ID=$(echo "$INGEST_RESPONSE" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['document_id'])")
echo "Document ID: $DOCUMENT_ID"
echo

# Grounded question
echo "--- Question 1: Grounded (interest rate) ---"
Q1_RESPONSE=$(curl -s -X POST "$BASE_URL/documents/$DOCUMENT_ID/questions" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the annual interest rate?"}')

pretty_print "$Q1_RESPONSE"
echo

# Unanswerable question
echo "--- Question 2: Unanswerable (CEO salary) ---"
Q2_RESPONSE=$(curl -s -X POST "$BASE_URL/documents/$DOCUMENT_ID/questions" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the CEO'\''s annual salary?"}')

pretty_print "$Q2_RESPONSE"
echo

echo "=== Demo Complete ==="
