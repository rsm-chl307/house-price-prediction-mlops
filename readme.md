#  End-to-End MLOps: House Price Prediction Pipeline
> **A complete machine learning lifecycle featuring automated data drift monitoring and experiment tracking.**

## 專案目標 (Project Objective)
本專案實作了一個具備「自我診斷能力」的 MLOps 生命週期。除了基礎的模型訓練與部署，本專案核心在於解決現實世界中最棘手的 **「數據偏移 (Data Drift)」** 問題。透過模擬時間序列資料流入，系統能自動偵測環境變遷（如通膨導致的房價結構改變），並確保模型在動態環境下的預測可靠性。

## 核心亮點 (Project Highlights)
* **數據守門員 (Data Gatekeeper)**：實作統計學 K-S 檢定，自動攔截並預警異常數據分佈。
* **數據與模型強耦合**：透過 MD5 Hashing 技術，確保每一份模型版本都能精準回溯至對應的數據指紋。
* **自動化健康報告**：Pipeline 執行後自動生成 Markdown 格式的模型健康與偏移報告。

## 技術棧與工具 (Tech Stack)
* **核心演算法**: XGBoost (Regression)
* **數據監控**: SciPy (Two-sample K-S Test)
* **實驗追蹤與註冊**: MLflow (託管於 DagsHub)
* **版本控制**: DVC (Data Version Control) + Git
* **自動化管線**: Bash Scripting (CI/CD Ready)
* **前端介面**: Streamlit

## 系統架構與自動化流程 (Pipeline Steps)

1.  **資料版本化與基準鎖定 (Baseline Initialization)**:
    使用 `init_baseline.py` 建立「黃金數據集 (Golden Dataset)」，作為後續監控的統計基準。
2.  **數據偏移檢測 (Data Drift Detection)**:
    在訓練前執行 `check_drift.py`。若新進資料與基準分佈差異過大（P-Value < 0.05），系統將自動發出預警並記錄於 `drift_report.json`。
3.  **實驗追蹤與數據指紋 (Tracking & Hashing)**:
    訓練過程中，MLflow 自動記錄超參數、RMSE 指標，並同步記錄該批次數據的 MD5 雜湊值，達成 100% 的實驗可重現性。
4.  **自動化管線整合 (Pipeline Automation)**:
    透過 `run_pipeline.sh` 一鍵執行從「數據更新 -> 偏移檢測 -> 模型訓練 -> 版本推送」的完整路徑。
5.  **即時監控報告 (Automated Reporting)**:
    管線末端自動生成 `PIPELINE_REPORT.md`，即時呈現目前的數據健康狀態與模型效能指標。

## 實驗結果展示 (Experiment Results)

| 階段 | 狀態 | 特徵分佈 (MedInc) | RMSE | 結論 |
| :--- | :--- | :--- | :--- | :--- |
| **Month 1** | 🟢 Stable | Baseline | 0.5361 | 系統初始標竿 |
| **Month 2** | ⚠️ Drifted | P-Value: 0.0000 | 0.5464 | 成功偵測偏移並重訓修正 |

## 未來展望 (Future Work)
- [ ] 整合 GitHub Actions 實現完全無人值守的 CI/CD/CT 閉環。
- [ ] 導入 Slack/Email Webhook 實現即時偏移告警通知。
