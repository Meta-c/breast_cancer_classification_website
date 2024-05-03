'use strict';


function previewImage(event) {
  const fileInput = event.target;
  const previewImage = document.querySelector('.preview-image');
  const predictButton = document.getElementById('predict-button');

  const file = fileInput.files[0];
  const reader = new FileReader();

  reader.onloadend = function() {
      previewImage.src = reader.result;
      // Show the predict button when file is selected
      predictButton.style.display = 'block';
  }

  if (file) {
      reader.readAsDataURL(file);
  } else {
      previewImage.src = "placeholder_image.png";
  }
}

function predictImage() {
  // Show loading modal
  const loadingModal = document.getElementById('loading-modal');
  loadingModal.style.display = 'block';

  var form = document.getElementById("upload-form");
  var formData = new FormData(form);

  fetch("/", {
      method: "POST",
      body: formData
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Network response was not ok');
      }
      return response.text();
  })
  .then(prediction => {
      // Hide loading modal
      loadingModal.style.display = 'none';

      // Display the prediction result
      document.getElementById("pred-box").innerHTML = "<h2 id='head-result'>Prediction Result <br /> <bre /> </h2>  <p id='result' >" + prediction  + "</p>";
  })
  .catch(error => {
      console.error('Error:', error);
      // Hide loading modal in case of error
      loadingModal.style.display = 'none';
  });

  // Set a fixed timeout to hide the loading modal after 5 seconds (adjust as needed)
  setTimeout(function() {
      loadingModal.style.display = 'none';
  }, 40000); // 5000 milliseconds = 5 seconds
}



document.addEventListener('DOMContentLoaded', function() {
  const links = document.querySelectorAll('a[href^="#"]');
  links.forEach(link => {
      link.addEventListener('click', function(e) {
          e.preventDefault();
          const targetId = this.getAttribute('href').substring(1);
          const targetElement = document.getElementById(targetId);
          if (targetElement) {
              const offsetTop = targetElement.offsetTop;
              window.scrollTo({
                  top: offsetTop,
                  behavior: 'smooth',
                  duration:1,
              });
          }
      });
  });
});
