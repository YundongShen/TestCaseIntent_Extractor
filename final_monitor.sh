#!/bin/bash
start=$(date +%s)
while true; do
  clear
  elapsed=$(($(date +%s) - start))
  mins=$((elapsed / 60))
  secs=$((elapsed % 60))

  echo "============================="
  echo "18-Job Monitor (elapsed: ${mins}m${secs}s)"
  echo "============================="
  echo ""

  running=$(squeue -u $USER -h -t running | wc -l)
  pending=$(squeue -u $USER -h -t pending | wc -l)

  echo "Status: running=$running  pending=$pending"
  echo ""

  extracts=$(ls -1 Result/extract_result/extract_result_20260420_*.json 2>/dev/null | wc -l)
  onboardings=$(ls -1 Result/onboarding_result/onboarding_20260420_*.json 2>/dev/null | wc -l)

  echo "Results generated:"
  echo "  Extract:    $extracts"
  echo "  Onboarding: $onboardings"
  echo ""

  if [ $running -eq 0 ] && [ $pending -eq 0 ]; then
    echo "All jobs finished."
    break
  fi

  sleep 3
done
