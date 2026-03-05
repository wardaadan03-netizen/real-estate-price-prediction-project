document.addEventListener('DOMContentLoaded', function() {
    // Add input validation
    const numberInputs = document.querySelectorAll('input[type="number"]');
    
    numberInputs.forEach(input => {
        input.addEventListener('change', function() {
            const min = parseFloat(this.min) || 0;
            const max = parseFloat(this.max) || Infinity;
            let value = parseFloat(this.value);
            
            if (isNaN(value)) {
                this.value = min;
            } else if (value < min) {
                this.value = min;
            } else if (value > max) {
                this.value = max;
            }
        });
    });
    
    // Add smooth scrolling
    document.querySelector('.predict-btn').addEventListener('click', function(e) {
        const result = document.querySelector('.prediction-result');
        if (result) {
            setTimeout(() => {
                result.scrollIntoView({ behavior: 'smooth' });
            }, 100);
        }
    });
    
    // Add tooltips
    const formGroups = document.querySelectorAll('.form-group');
    formGroups.forEach(group => {
        const input = group.querySelector('input, select');
        const label = group.querySelector('label');
        
        if (input && label) {
            input.setAttribute('placeholder', label.textContent);
        }
    });
    
    // Optional: Add API call for real-time validation
    const postcodeInput = document.getElementById('postcode');
    if (postcodeInput) {
        postcodeInput.addEventListener('blur', function() {
            const postcode = this.value;
            // Validate Melbourne postcodes (3000-3999 for Victoria)
            if (postcode < 3000 || postcode > 3999) {
                alert('Warning: Postcode may not be in Melbourne metropolitan area');
            }
        });
    }
});