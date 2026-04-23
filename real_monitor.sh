#!/bin/bash

echo "=========================================="
echo "Monitoring 18 job execution progress"
echo "Start time: $(date)"
echo "=========================================="
echo ""

while true; do
  clear
  timestamp=$(date '+%H:%M:%S')
  echo "=========================================="
  echo "Job Monitor - $timestamp"
  echo "=========================================="

  running=$(squeue -u $USER -h -t running | wc -l)
  pending=$(squeue -u $USER -h -t pending | wc -l)

  echo "Status:"
  echo "  Running: $running"
  echo "  Pending: $pending"
  echo ""

  echo "Queue (first 5):"
  squeue -u $USER | head -6
  echo ""

  extract_count=$(ls -1 Result/extract_result/*20260420* 2>/dev/null | tail -18 | wc -l)
  onboarding_count=$(ls -1 Result/onboarding_result/*20260420* 2>/dev/null | tail -18 | wc -l)

  echo "Results (last 18):"
  echo "  Extract:    $extract_count"
  echo "  Onboarding: $onboarding_count"
  echo ""

  if [ "$running" -eq 0 ] && [ "$pending" -eq 0 ]; then
    echo "All jobs completed."
    break
  fi

  echo "Next update in 5s..."
  sleep 5
done
