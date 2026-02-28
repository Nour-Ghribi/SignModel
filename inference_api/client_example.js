const API_URL = "http://localhost:8000/predict";

export async function predictSign(sequence) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sequence }),
  });

  if (!response.ok) {
    throw new Error(`Predict failed: ${response.status}`);
  }

  return response.json();
}

// Example:
// const result = await predictSign(sequence16x126);
// console.log(result.label, result.score, result.top3);
