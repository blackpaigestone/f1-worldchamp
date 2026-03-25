coef_df = pd.DataFrame({
    "feature": features_cls_core,
    "coef": logit_x.coef_[0]
}).sort_values(by="coef", ascending=False)

coef_df

coef_df["odds_ratio"] = np.exp(coef_df["coef"])
coef_df

prob_true, prob_pred = calibration_curve(y_test_x, probs_x, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.title("Calibration Curve — DNF Model")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.legend()
plt.grid(True)
plt.show()

lift_df = test.copy()
lift_df["pred_x_prob"] = probs_x

lift_df["dnf_decile"] = pd.qcut(
    lift_df["pred_x_prob"],
    q=10,
    labels=False,
    duplicates="drop"
) + 1

decile_perf = (
    lift_df.groupby("dnf_decile")
    .agg(
        actual_x_rate=("target_x", "mean"),
        avg_pred_prob=("pred_x_prob", "mean"),
        n=("target_x", "size")
    )
    .reset_index()
)

plt.figure(figsize=(8, 5))
plt.plot(decile_perf["dnf_decile"], decile_perf["actual_x_rate"], marker="o", label="Actual")
plt.plot(decile_perf["dnf_decile"], decile_perf["avg_pred_prob"], marker="s", linestyle="--", label="Predicted")
plt.title("Lift by Prediction Decile — DNF")
plt.xlabel("Prediction Decile")
plt.ylabel("DNF Rate / Predicted Probability")
plt.legend()
plt.grid(True)
plt.show()

xgb_x = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_x.fit(X_train, y_train_x)

probs_x_xgb = xgb_x.predict_proba(X_test)[:, 1]
auc_x_xgb = roc_auc_score(y_test_x, probs_x_xgb)

auc_x_xgb

model_compare = pd.DataFrame({
    "model": ["Logistic Regression", "XGBoost"],
    "roc_auc": [auc_x, auc_x_xgb]
})

model_compare

explainer = shap.Explainer(xgb_x)
shap_values = explainer(X_test)

shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)

pit_effect = (
    df.groupby("slow_pit_stop")["target_x"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(6, 4))
sns.barplot(data=pit_effect, x="slow_pit_stop", y="target_x")
plt.title("DNF Rate: Slow vs Fast Pit Stops")
plt.xlabel("Slow Pit Stop")
plt.ylabel("DNF Rate")
plt.show()

driver_pit = (
    df.groupby(["driverId", "slow_pit_stop"])["target_x"]
    .mean()
    .reset_index())

team_pit = (
    df.groupby(["constructorId", "slow_pit_stop"])["target_x"]
    .mean()
    .reset_index()
)

team_pit.head()

driver_pit = (
    df.groupby(["driverId", "slow_pit_stop"])["target_x"]
    .mean()
    .reset_index()
)

driver_pit.head()