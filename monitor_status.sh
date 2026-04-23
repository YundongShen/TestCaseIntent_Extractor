#!/bin/bash

echo "=========================================="
echo "Real-time monitor for 18 jobs"
echo "Start time: $(date)"
echo "=========================================="

while true; do
  clear
  echo "=========================================="
  echo "Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="
  echo ""

  running=$(squeue -u $USER -h | wc -l)
  echo "Jobs currently running: $running"

  if [ $running -gt 0 ]; then
    echo "Queue:"
    squeue -u $USER -h | head -5
  fi

  echo ""
  echo "Results generated:"
  extract_count=$(ls -1 Result/extract_result/extract_result_*.json 2>/dev/null | wc -l)
  onboarding_count=$(ls -1 Result/onboarding_result/onboarding_*.md 2>/dev/null | wc -l)

  echo "  Extract:    $extract_count"
  echo "  Onboarding: $onboarding_count"
  echo ""

  if [ $extract_count -gt 0 ]; then
    echo "Latest file:"
    ls -lt Result/extract_result/extract_result_*.json 2>/dev/null | head -1
  fi

  echo ""

  if [ $running -eq 0 ]; then
    if [ $extract_count -eq 18 ] && [ $onboarding_count -eq 18 ]; then
      echo "All 18 jobs completed."
      break
    elif [ $extract_count -gt 0 ]; then
      echo "Jobs stopped — results incomplete ($extract_count/18)"
      break
    fi
  fi

  echo "Next update in 10s"
  sleep 10
done

echo ""
echo "=========================================="
echo "Final summary:"
final_extract=$(ls -1 Result/extract_result/extract_result_*.json 2>/dev/null | wc -l)
final_onboarding=$(ls -1 Result/onboarding_result/onboarding_*.md 2>/dev/null | wc -l)
echo "  Extract results:    $final_extract"
echo "  Onboarding results: $final_onboarding"
echo "=========================================="
