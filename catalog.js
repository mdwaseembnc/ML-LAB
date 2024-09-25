const fs = require('fs');

// Function to perform Lagrange Interpolation to find f(0)
function lagrangeInterpolation(points) {
    let result = 0;

    // Lagrange interpolation formula to calculate f(0)
    for (let i = 0; i < points.length; i++) {
        let term = points[i].y;

        for (let j = 0; j < points.length; j++) {
            if (i !== j) {
                term *= (-points[j].x) / (points[i].x - points[j].x);
            }
        }
        result += term;
    }
    return result;
}

// Function to decode the y value based on the given base
function decodeYValue(encodedY, base) {
    return parseInt(encodedY, base);
}

// Main function to read the input JSON, decode y values, and calculate the constant term
function findConstantTermFromJsonFile(filePath) {
    try {
        // Read and parse the input JSON
        const data = fs.readFileSync(filePath, 'utf8');
        const jsonInput = JSON.parse(data);

        const n = jsonInput.keys.n;
        const k = jsonInput.keys.k;

        // Prepare the points (x, y)
        let points = [];
        let keys = Object.keys(jsonInput).filter(key => key !== "keys");

        for (let i = 0; i < keys.length; i++) {
            const key = keys[i];
            const base = parseInt(jsonInput[key].base);
            const value = jsonInput[key].value;

            // Decode y value from the given base
            const y = decodeYValue(value, base);

            // Push the point (x, y) where x is the key, and y is the decoded value
            points.push({
                x: parseInt(key),
                y: y
            });

            // Stop after collecting k points (minimum required points to solve the polynomial)
            if (points.length >= k) break;
        }

        // Calculate the constant term using Lagrange interpolation
        const constantTerm = lagrangeInterpolation(points);

        // Output the constant term
        console.log(`The constant term c (f(0)) is: ${constantTerm.toFixed(6)}`);
    } catch (err) {
        console.error('Error:', err.message);
    }
}

// Provide the path to the input JSON file
const filePath = './input.json';  // Modify this path as needed
findConstantTermFromJsonFile(filePath);
