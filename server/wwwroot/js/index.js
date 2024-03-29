function uploadFile() {
  const fileInput = document.getElementById("fileInput");

  if (fileInput.files.length === 0) {
    alert("Faili tahaks kangesti");
    return;
  }

  const formData = new FormData();
  formData.append("File", fileInput.files[0]);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      const resultContainer = document.getElementById("resultContainer");
      resultContainer.innerHTML =
        '<div class="col-12"><h2>Ennustus:</h2>' +
        "Tõenäosus, et on ehitud: " +
        (data.decorated * 100).toFixed(2) +
        '% </div class="col-12">' +
        "<div>Tõenäosus, et on ehtimata: " +
        (data.undecorated * 100).toFixed(2) +
        "% </div>";
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Ikaldus. Ei tea.");
    });
}
