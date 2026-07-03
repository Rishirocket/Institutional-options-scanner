# Options Telegram Website Bot

Website + GitHub Actions scanner for SPY, QQQ, and liquid large-cap option alerts.

## What it does
- Scans 7DTE, 30DTE, and 90DTE option setups
- Uses monthly EMA50 bias, daily trend, 4H EMA stack, volume profile shelves, IV/HV, and option liquidity
- Sends Telegram alerts for CALL/PUT setups scoring 70/100+
- Includes a small Flask dashboard

## GitHub setup
1. Upload this folder to a new GitHub repo.
2. Go to **Settings → Secrets and variables → Actions → New repository secret**.
3. Add:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. Go to **Actions → Options Telegram Website Bot → Run workflow**.

## Local website
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Scheduled alerts
The workflow runs hourly:
```yaml
cron: "0 * * * *"
```

## Test Telegram
Temporarily change this in `.github/workflows/options_bot.yml`:
```yaml
SEND_TEST: "1"
```
Then run workflow manually. Change it back to `0` after you receive the test.

## Risk note
This creates setup alerts, not guaranteed buy/sell signals. Verify fill, spread, news, earnings, volume, and chart before trading.
