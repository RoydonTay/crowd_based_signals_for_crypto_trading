navigating this folder:

notebooks:
1. topic_modelling_oneshot.ipynb
- notebook for one shot assigning a buy/hold/sell signal classification to each post/comment before aggregating the classifications by day and assigning a signal to a day
- classification done for each cryptocurrency (eth/btc)
- generated daily signals are saved to predicted_signals folder (each file: <coinName>_<model>_daily_signals.csv)

2. signal_evaluation.ipynb
- loads actual btc/eth daily prices and merges the prices with the generated daily signals for eyeball comparison

3. actual_topic_modelling.ipynb (a work in progress)
- an attempt at performing unsupervised topic modeling to discover 5-8 key discussion themes
- generates daily topic distribution features by aggregating topic probabilities for each day
- outputs from this notebook is saved to actual_topic_modelled

folders:
# scraped reddit dataset
1. BTC
2. ETH
- copies of the shared dataset folders 
# processed datasets
3. extracted_datasets
- <coinName>_reddit_data.csv: datasets merged from the separate files from the BTC/ETH folders
- <coinName>_preprocessed.csv: dataset that's been preprocessed for one shot signal classifcation (from topic_modelling_oneshot.ipynb)
4. predicted_signals
- <coinName>_<model>_daily_signals.csv: daily signals predicted from topic_modelling_oneshot.ipynb (columns: date,daily_signal,signal_strength,buy_ratio,sell_ratio,hold_ratio,total_posts,avg_confidence,total_score,model_used)
- <coinName>_<model>_signal_df.csv: full dataset from extracted_datasets, just with additional columns containing the predicted signal and score etc. (also from topic_modelling_oneshot.ipynb) <coinName>_<model>_daily_signals.csv is derived from this dataframe
5. actual_topic_modelled
- just a folder to dump relevant dataframes generated from actual_topic_modelling.ipynb (a work in progress!)