#!/bin/bash

# Monitor progress of 18 jobs

echo "=========================================="
echo "Job Progress Monitor"
echo "Start time: $(date)"
echo "=========================================="
echo ""

while true; do
  clear
  echo "=========================================="
  echo "Job Progress Monitor - $(date +%H:%M:%S)"
  echo "=========================================="
  echo ""

  running=$(squeue -u $USER -h -t running | wc -l)
  pending=$(squeue -u $USER -h -t pending | wc -l)
  completed=$(squeue -u $USER -h -t completed | wc -l)
  failed=$(squeue -u $USER -h -t failed | wc -l)

  echo "Job status:"
  echo "  Running:   $running"
  echo "  Pending:   $pending"
  echo "  Completed: $completed"
  echo "  Failed:    $failed"
  echo ""

  echo "Queue (first 20):"
  squeue -u $USER | head -21
  echo ""

  extract_count=$(ls -1 Result/extract_result/*20260420* 2>/dev/null | wc -l)
  onboarding_count=$(ls -1 Result/onboarding_result/*20260420* 2>/dev/null | wc -l)

  echo "Result files:"
  echo "  Extract:    $extract_count"
  echo "  Onboarding: $onboarding_count"
  echo ""

  total_jobs=$(squeue -u $USER | grep -c "test_")
  if [ $total_jobs -eq 0 ]; then
    echo "All jobs completed."
    echo "Final count:"
    final_extract=$(ls -1 Result/extract_result/*20260420* 2>/dev/null | wc -l)
    final_onboarding=$(ls -1 Result/onboarding_result/*20260420* 2>/dev/null | wc -l)
    echo "  Extract:    $final_extract"
    echo "  Onboarding: $final_onboarding"
    break
  fi

  echo "Next update in 5s"
  sleep 5
done
