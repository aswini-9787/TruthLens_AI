
// Get HTML elements
const newsInput = document.getElementById("newsInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const result = document.getElementById("result");
const resultText = document.getElementById("resultText");

// Analyze News Button
analyzeBtn.addEventListener("click", function () {

    const news = newsInput.value.trim();

    // Check empty input
    if (news === "") {
        resultText.textContent = "Please enter a news article first.";
        return;
    }

    // Send news to Flask backend
    resultText.textContent = "Analyzing news...";

    fetch("/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            news: news
        })
    })

    .then(response => response.json())

    .then(data => {
        if (data.error) {
            resultText.textContent = data.error;
        } else {
            resultText.textContent =
                "Prediction: " + data.prediction +
                " | Confidence: " + data.confidence + "%";
        }
    })

    .catch(error => {
        resultText.textContent =
            "Unable to connect to the server. Please try again.";
        console.error(error);
    });

});

// Clear Button
clearBtn.addEventListener("click", function () {

    newsInput.value = "";

    resultText.textContent =
        "Your analysis result will appear here.";

});
