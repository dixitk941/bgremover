document.addEventListener('DOMContentLoaded', function() {
    // Upload box functionality
    const uploadBox = document.querySelector('.upload-container');
    const fileInput = document.getElementById('image-upload');
    
    if (uploadBox && fileInput) {
        uploadBox.addEventListener('click', function() {
            fileInput.click();
        });
        
        uploadBox.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadBox.classList.add('drag-over');
        });
        
        uploadBox.addEventListener('dragleave', function() {
            uploadBox.classList.remove('drag-over');
        });
        
        uploadBox.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadBox.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
        
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length) {
                handleFileUpload(fileInput.files[0]);
            }
        });
    }
    
    function handleFileUpload(file) {
        // In a real application, you would handle the file upload to your server here
        // For demo purposes, we'll just show an alert
        alert(`File "${file.name}" selected! In a real app, this would be processed to remove the background.`);
        
        // Redirect to the processing page after a short delay
        // setTimeout(() => {
        //     window.location.href = '/processing.html';
        // }, 1500);
    }
    
    // FAQ Accordion
    const faqItems = document.querySelectorAll('.faq-item');
    
    if (faqItems.length) {
        faqItems.forEach(item => {
            const question = item.querySelector('.faq-question');
            
            question.addEventListener('click', () => {
                const isActive = item.classList.contains('active');
                
                // Close all items
                faqItems.forEach(faq => {
                    faq.classList.remove('active');
                });
                
                // Open clicked item if it wasn't already active
                if (!isActive) {
                    item.classList.add('active');
                }
            });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Before-After Slider Interaction
    const slider = document.querySelector('.before-after-slider');
    
    if (slider) {
        let isDown = false;
        let startX;
        let scrollLeft;
        
        slider.addEventListener('mousedown', (e) => {
            isDown = true;
            startX = e.pageX;
            scrollLeft = slider.scrollLeft;
        });
        
        slider.addEventListener('mouseleave', () => {
            isDown = false;
        });
        
        slider.addEventListener('mouseup', () => {
            isDown = false;
        });
        
        slider.addEventListener('mousemove', (e) => {
            if (!isDown) return;
            e.preventDefault();
            
            const x = e.pageX;
            const walk = (x - startX);
            const sliderWidth = slider.offsetWidth;
            const percentage = (scrollLeft + walk) / sliderWidth * 100;
            
            // Constrain to between 0 and 100%
            const clampedPercentage = Math.max(0, Math.min(100, percentage));
            
            // Update clip path
            const before = slider.querySelector('.before');
            before.style.clipPath = `polygon(0 0, ${clampedPercentage}% 0, ${clampedPercentage}% 100%, 0 100%)`;
        });
        
        // Touch events for mobile
        slider.addEventListener('touchstart', (e) => {
            isDown = true;
            startX = e.touches[0].pageX;
            scrollLeft = slider.scrollLeft;
        });
        
        slider.addEventListener('touchend', () => {
            isDown = false;
        });
        
        slider.addEventListener('touchmove', (e) => {
            if (!isDown) return;
            
            const x = e.touches[0].pageX;
            const walk = (x - startX);
            const sliderWidth = slider.offsetWidth;
            const percentage = (scrollLeft + walk) / sliderWidth * 100;
            
            // Constrain to between 0 and 100%
            const clampedPercentage = Math.max(0, Math.min(100, percentage));
            
            // Update clip path
            const before = slider.querySelector('.before');
            before.style.clipPath = `polygon(0 0, ${clampedPercentage}% 0, ${clampedPercentage}% 100%, 0 100%)`;
        });
    }
});


document.addEventListener('DOMContentLoaded', function() {
    // Scroll animation for elements
    const fadeElements = document.querySelectorAll('section h2, .feature-card, .step, .pricing-card');
    
    // Add the fade-in-element class to all elements we want to animate
    fadeElements.forEach(element => {
        element.classList.add('fade-in-element');
    });
    
    // Function to check if an element is in viewport
    function isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top <= (window.innerHeight || document.documentElement.clientHeight) * 0.8
        );
    }
    
    // Function to handle scroll and animate elements
    function handleScroll() {
        fadeElements.forEach(element => {
            if (isInViewport(element)) {
                element.classList.add('visible');
            }
        });
    }
    
    // Call once on load
    handleScroll();
    
    // Add scroll event listener
    window.addEventListener('scroll', handleScroll);
    
    // Counter animation for statistics
    function animateCounter(element, target, duration = 2000) {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target.toLocaleString();
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current).toLocaleString();
            }
        }, 16);
    }
    
    // Animate the hero image sequence
    const heroImage = document.querySelector('.image-animation');
    if (heroImage) {
        heroImage.classList.add('animate');
    }
    
    // Type writer effect for title
    function typeWriter(element, text, speed = 50, delay = 0) {
        let i = 0;
        
        setTimeout(() => {
            const interval = setInterval(() => {
                if (i < text.length) {
                    element.innerHTML += text.charAt(i);
                    i++;
                } else {
                    clearInterval(interval);
                }
            }, speed);
        }, delay);
    }
    
    // Animate the before-after slider on hover
    const beforeAfterSlider = document.querySelector('.before-after-slider');
    
    if (beforeAfterSlider) {
        beforeAfterSlider.addEventListener('mouseover', () => {
            const before = beforeAfterSlider.querySelector('.before');
            
            // Create a CSS animation to move the slider back and forth
            before.style.animation = 'sliderAnimation 3s ease-in-out forwards';
            
            // Remove the animation after it completes
            setTimeout(() => {
                before.style.animation = '';
            }, 3000);
        });
    }
    
    // Add keyframes for the slider animation dynamically
    const style = document.createElement('style');
    style.textContent = `
        @keyframes sliderAnimation {
            0% { clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%); }
            50% { clip-path: polygon(0 0, 80% 0, 80% 100%, 0 100%); }
            100% { clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%); }
        }
    `;
    document.head.appendChild(style);
    
    // Parallax effect for sections
    window.addEventListener('scroll', () => {
        const scrollPosition = window.pageYOffset;
        
        document.querySelectorAll('.parallax').forEach(element => {
            const speed = element.getAttribute('data-speed') || 0.5;
            element.style.transform = `translateY(${scrollPosition * speed}px)`;
        });
    });
    
    // Initialize the interactive demos
    initializeAnimatedDemos();
});

// Function to initialize the animated demos
function initializeAnimatedDemos() {
    // This function would set up any interactive demos
    // In a real implementation, you might want to connect to your backend API
    // or use a service like Cloudinary to do the actual background removal
    
    console.log('Animated demos initialized');
}