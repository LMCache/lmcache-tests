import pandas as pd
import numpy as np

# Read CSV
file_path = 'test_lmcache_redis_sentinel.csv'  
data = pd.read_csv(file_path)

# Calculate ITL (1 / throughput)
data['ITL'] = 1 / data['throughput']

# Separate data for engine_id = 0 and engine_id = 1
engine_0 = data[data['engine_id'] == 0]
engine_1 = data[data['engine_id'] == 1]

# Calculate aggregated metrics for engine_id = 0
cache_avg_latency = engine_0['latency'].mean()
cache_p90_latency = np.percentile(engine_0['latency'], 90)
cache_avg_ttft = engine_0['TTFT'].mean()
cache_avg_itl = engine_0['ITL'].mean()

# Calculate aggregated metrics for engine_id = 1
wocache_avg_latency = engine_1['latency'].mean()
wocache_p90_latency = np.percentile(engine_1['latency'], 90)
wocache_avg_ttft = engine_1['TTFT'].mean()
wocache_avg_itl = engine_1['ITL'].mean()

# Create final summarized row
summary = {
    "Concurrency": 1,  # Assume constant concurrency for this scenario
    "Cache Avg. Latency": cache_avg_latency,
    "Cache P90 Latency": cache_p90_latency,
    "Cache Avg. TTFT": cache_avg_ttft,
    "Cache Avg. ITL": cache_avg_itl,
    "W/o Cache Avg. Latency": wocache_avg_latency,
    "W/o Cache P90 Latency": wocache_p90_latency,
    "W/o Cache Avg. TTFT": wocache_avg_ttft,
    "W/o Cache Avg. ITL": wocache_avg_itl,
}

# Convert to DataFrame and save as CSV
summary_df = pd.DataFrame([summary])
output_path = 'redis_summary_one_row.csv'
summary_df.to_csv(output_path, index=False)

print("Table generated and saved to:", output_path)
