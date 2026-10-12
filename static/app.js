/**
 * Apple Quality AI - Frontend Interactive Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // 1. Elements & References
    // -------------------------------------------------------------
    const features = ["size", "weight", "sweetness", "crunchiness", "juiciness", "ripeness", "acidity"];
    
    const sliders = {};
    const badges = {};
    features.forEach(feat => {
        sliders[feat] = document.getElementById(`feature-${feat}`);
        badges[feat] = document.getElementById(`val-badge-${feat}`);
    });

    const form = document.getElementById("prediction-form");
    const modelRadios = document.querySelectorAll('input[name="model-choice"]');

    // UI Decision Elements
    const heroCard = document.getElementById("decision-hero");
    const badgeIcon = document.getElementById("badge-icon");
    const badgeTitle = document.getElementById("badge-title");
    const badgeSubtitle = document.getElementById("badge-subtitle");
    const primaryConfValue = document.getElementById("primary-conf-value");
    const confFillBar = document.getElementById("conf-fill-bar");
    const goodProbText = document.getElementById("good-prob-text");
    const badProbText = document.getElementById("bad-prob-text");

    // Duel Model Elements
    const knnPredPill = document.getElementById("knn-pred-pill");
    const knnConfLabel = document.getElementById("knn-conf-label");
    const knnBarFill = document.getElementById("knn-bar-fill");

    const nbPredPill = document.getElementById("nb-pred-pill");
    const nbConfLabel = document.getElementById("nb-conf-label");
    const nbBarFill = document.getElementById("nb-bar-fill");

    const agreementBanner = document.getElementById("agreement-banner");
    const agreementText = document.getElementById("agreement-text");
    const profileTagsContainer = document.getElementById("profile-tags");

    // -------------------------------------------------------------
    // 2. Tab Navigation
    // -------------------------------------------------------------
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            tabButtons.forEach(b => {
                b.classList.remove("active");
                b.setAttribute("aria-selected", "false");
            });
            tabPanes.forEach(pane => pane.classList.remove("active"));

            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");
            const activePane = document.getElementById(`tab-${targetTab}`);
            if (activePane) {
                activePane.classList.add("active");
            }
        });
    });

    // -------------------------------------------------------------
    // 3. Slider Binding & Live Value Updates
    // -------------------------------------------------------------
    let debounceTimer = null;

    function getSelectedModel() {
        const checked = document.querySelector('input[name="model-choice"]:checked');
        return checked ? checked.value : "both";
    }

    function updateBadge(feat) {
        if (sliders[feat] && badges[feat]) {
            const val = parseFloat(sliders[feat].value);
            badges[feat].textContent = (val >= 0 ? "+" : "") + val.toFixed(2);
        }
    }

    features.forEach(feat => {
        const slider = sliders[feat];
        if (!slider) return;

        updateBadge(feat);

        slider.addEventListener("input", () => {
            updateBadge(feat);
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                runPrediction();
            }, 120);
        });
    });

    // Model choice change listener
    modelRadios.forEach(radio => {
        radio.addEventListener("change", () => {
            document.querySelectorAll(".model-pill").forEach(p => p.classList.remove("active"));
            radio.closest(".model-pill").classList.add("active");
            runPrediction();
        });
    });

    // Form submission listener
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        runPrediction();
    });

    // -------------------------------------------------------------
    // 4. API Request & Prediction Handling
    // -------------------------------------------------------------
    async function runPrediction() {
        const payload = {
            model: getSelectedModel(),
            Size: parseFloat(sliders.size.value),
            Weight: parseFloat(sliders.weight.value),
            Sweetness: parseFloat(sliders.sweetness.value),
            Crunchiness: parseFloat(sliders.crunchiness.value),
            Juiciness: parseFloat(sliders.juiciness.value),
            Ripeness: parseFloat(sliders.ripeness.value),
            Acidity: parseFloat(sliders.acidity.value)
        };

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();
            renderPredictionResults(data, payload);
        } catch (err) {
            console.error("Prediction failed:", err);
        }
    }

    function renderPredictionResults(data, featuresPayload) {
        const isGood = data.primary_prediction.toLowerCase() === "good";
        const primaryConf = data.primary_confidence;

        // 1. Update Hero Card
        if (isGood) {
            heroCard.classList.remove("verdict-bad");
            heroCard.classList.add("verdict-good");
            badgeIcon.textContent = "🍏";
            badgeTitle.textContent = "GOOD QUALITY";
            badgeSubtitle.textContent = "Crisp, sweet, and optimal sensory profile.";
        } else {
            heroCard.classList.remove("verdict-good");
            heroCard.classList.add("verdict-bad");
            badgeIcon.textContent = "🍂";
            badgeTitle.textContent = "BAD QUALITY";
            badgeSubtitle.textContent = "Sub-optimal acidity, sweetness, or texture defect detected.";
        }

        // 2. Confidence Fill & Probabilities
        primaryConfValue.textContent = `${primaryConf}%`;
        confFillBar.style.width = `${primaryConf}%`;

        const goodProb = data.knn.prob_good;
        const badProb = data.knn.prob_bad;
        goodProbText.textContent = `${goodProb}%`;
        badProbText.textContent = `${badProb}%`;

        // 3. KNN Model Box
        const knnGood = data.knn.prediction.toLowerCase() === "good";
        knnPredPill.textContent = data.knn.prediction.toUpperCase();
        knnPredPill.className = `duel-pred-pill ${knnGood ? "" : "pred-bad"}`;
        knnConfLabel.textContent = `${data.knn.confidence}% Conf`;
        knnBarFill.style.width = `${data.knn.confidence}%`;
        knnBarFill.style.backgroundColor = knnGood ? "var(--primary-emerald)" : "var(--primary-rose)";

        // 4. Naive Bayes Model Box
        const nbGood = data.naive_bayes.prediction.toLowerCase() === "good";
        nbPredPill.textContent = data.naive_bayes.prediction.toUpperCase();
        nbPredPill.className = `duel-pred-pill ${nbGood ? "" : "pred-bad"}`;
        nbConfLabel.textContent = `${data.naive_bayes.confidence}% Conf`;
        nbBarFill.style.width = `${data.naive_bayes.confidence}%`;
        nbBarFill.style.backgroundColor = nbGood ? "var(--primary-emerald)" : "var(--primary-rose)";

        // 5. Consensus / Agreement Banner
        if (data.knn.prediction === data.naive_bayes.prediction) {
            agreementBanner.className = "agreement-banner";
            agreementBanner.querySelector(".agreement-icon").textContent = "✓";
            agreementText.textContent = `Both models agree on ${data.knn.prediction.toUpperCase()} quality verdict.`;
        } else {
            agreementBanner.className = "agreement-banner disagree";
            agreementBanner.querySelector(".agreement-icon").textContent = "⚠";
            agreementText.textContent = `Models split! KNN predicts ${data.knn.prediction.toUpperCase()} (KNN accuracy is 13.5% higher).`;
        }

        // 6. Dynamic Sensory Profile Tags
        updateSensoryTags(featuresPayload);
    }

    function updateSensoryTags(p) {
        const tags = [];
        if (p.Sweetness > 1.5) tags.push("🍯 Very Sweet");
        else if (p.Sweetness < -1.5) tags.push("🍋 Low Sweetness");

        if (p.Crunchiness > 1.2) tags.push("⚡ Crisp & Crunchy");
        else if (p.Crunchiness < -1.2) tags.push("🍂 Mealy / Soft Texture");

        if (p.Juiciness > 1.2) tags.push("💧 High Moisture");
        else if (p.Juiciness < -1.2) tags.push("🏜 Dry Flesh");

        if (p.Ripeness > 2.0) tags.push("⚠️ Overripe Risk");
        else if (p.Ripeness < -1.5) tags.push("🌱 Unripe Flesh");

        if (p.Acidity > 2.0) tags.push("⚡ Sharp Acidity");

        if (tags.length === 0) tags.push("⚖️ Balanced Physical Attributes");

        profileTagsContainer.innerHTML = tags
            .map(t => `<span class="profile-tag">${t}</span>`)
            .join("");
    }

    // -------------------------------------------------------------
    // 5. Presets Handling
    // -------------------------------------------------------------
    function setValues(valMap) {
        features.forEach(feat => {
            const key = feat.charAt(0).toUpperCase() + feat.slice(1);
            const val = valMap[key] !== undefined ? valMap[key] : (valMap[feat] !== undefined ? valMap[feat] : 0.0);
            if (sliders[feat]) {
                sliders[feat].value = val;
                updateBadge(feat);
            }
        });
        runPrediction();
    }

    // Preset 1: Premium Crisp Sweet
    document.getElementById("btn-preset-premium").addEventListener("click", () => {
        setValues({
            Size: 1.8,
            Weight: 0.9,
            Sweetness: 3.5,
            Crunchiness: 2.4,
            Juiciness: 2.7,
            Ripeness: -0.4,
            Acidity: -0.7
        });
    });

    // Preset 2: Overripe Soft
    document.getElementById("btn-preset-spoiled").addEventListener("click", () => {
        setValues({
            Size: -2.9,
            Weight: -2.3,
            Sweetness: -2.5,
            Crunchiness: -3.6,
            Juiciness: -1.8,
            Ripeness: 4.8,
            Acidity: 2.1
        });
    });

    // Preset 3: Random Sample from Dataset API
    document.getElementById("btn-preset-random").addEventListener("click", async () => {
        try {
            const res = await fetch("/api/sample");
            if (!res.ok) throw new Error("Sample fetch failed");
            const row = await res.json();
            setValues(row);
        } catch (e) {
            console.error("Could not fetch random sample:", e);
        }
    });

    // Reset Sliders
    document.getElementById("btn-reset-sliders").addEventListener("click", () => {
        setValues({
            Size: 0.0,
            Weight: 0.0,
            Sweetness: 0.0,
            Crunchiness: 0.0,
            Juiciness: 0.0,
            Ripeness: 0.0,
            Acidity: 0.0
        });
    });

    // -------------------------------------------------------------
    // 6. FAQ Accordion Toggling
    // -------------------------------------------------------------
    document.querySelectorAll(".faq-item").forEach(item => {
        const questionBtn = item.querySelector(".faq-question");
        questionBtn.addEventListener("click", () => {
            const isOpen = item.classList.contains("open");
            // Close other items
            document.querySelectorAll(".faq-item").forEach(other => other.classList.remove("open"));
            if (!isOpen) {
                item.classList.add("open");
            }
        });
    });

    // Run initial prediction on load
    runPrediction();
});
