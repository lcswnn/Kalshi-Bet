# KALSHI BET

## How to use program
1. git clone project to a folder of your choice
2. Navigate into the folder, and navigate into Kalshi-Bet
3. Once in the Kalshi-Bet folder, create a new virtual enviornment.
   - Mac Users run: `python3 -m venv venv`
   - Windows Users run: ``
4. Activate the venv you just created using the command:
   - Mac/Linux: `source venv/bin/activate`
   - Windows:``
5. Inside your venv (you should see `(venv)` at the beginning of your command lines now), install the requirements.txt file doing:
   - Mac/Linux: `pip3 install -r requirements.txt`
   - Windows:``
6. After that, run the model you want. For instance, if you want to run the kalshi_model_ensemble.py, run:
   - Mac/Linux: `python3 kalshi_model_ensemble.py`
   - Windows: ``
7. Enjoy the generational wealth.

## Stats
- According to the backtesting python file, we are reaching a 25.6% Return On Investment (ROI)
- We are winning 65.9% of the bets we are making
   - Notes on Backtesting:
   - This backtest uses simulated forecasts (actual temp + noise).
   - Real performance depends on actual forecast quality.
- We begin by using Half Kelly Criterion, then going from there once we establish Ensemble_V8 winning statistics
- Model has a sweet spot of stats that we want to hit, making sure that the prices are within a range we can maximize our edge within.
- If there is a price that is too low, we don't bet on it, as well as a price that is not too high or within our sweet spot of betting.
  



