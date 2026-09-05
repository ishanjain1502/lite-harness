-- stop-reason histogram
SELECT stop_reason, count() AS n
FROM liteness.session_events FINAL
WHERE event_type = 'turn/end'
GROUP BY stop_reason
ORDER BY n DESC;

-- tool error rate
SELECT tool_name,
       countIf(is_error = 1) / count() AS error_rate,
       count() AS calls
FROM liteness.session_events FINAL
WHERE event_type = 'tool/result'
GROUP BY tool_name
ORDER BY error_rate DESC;

-- orphan tool/call (in calls, not in results)
SELECT session_id, call_id
FROM liteness.session_events FINAL
WHERE event_type = 'tool/call' AND call_id != ''
  AND (session_id, call_id) NOT IN (
    SELECT session_id, call_id
    FROM liteness.session_events FINAL
    WHERE event_type = 'tool/result' AND call_id != ''
  );

-- eval pass rate
SELECT suite, evaluator, avg(passed) AS pass_rate, count() AS n
FROM liteness.eval_results FINAL
GROUP BY suite, evaluator;

-- sessions with ledger rows and no ops/shutdown
SELECT DISTINCT session_id
FROM liteness.session_events FINAL
WHERE channel = 'ledger'
  AND session_id NOT IN (
    SELECT session_id
    FROM liteness.session_events FINAL
    WHERE event_type = 'ops/shutdown'
  );

-- JSONL vs warehouse: compare len(session.events) minus extra assistant/chunk
-- to count() WHERE session_id = {id} AND channel = 'ledger'
