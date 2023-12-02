function uploadFile() {
    const fileInput = document.getElementById('fileInput');
  
    if (fileInput.files.length === 0) {
      alert('Faili tahaks kangesti');
      return;
    }

    const formData = new FormData();
    formData.append('File', fileInput.files[0]);

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      const resultContainer = document.getElementById('resultContainer');
      resultContainer.innerHTML = '<h2>Ennustus:</h2><pre>' + JSON.stringify(data, null, 2) + '</pre>';
    })
    .catch(error => {
      console.error('Error:', error);
      alert('Ikaldus. Ei tea.');
    });
  }