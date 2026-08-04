const express = require('express');
const router = express.Router();
const { getInsights, getLatestPrediction } = require('../controllers/InsightsController');

router.get('/', getInsights);

// Get latest prediction for a specific symbol
router.get('/latest/:symbol', getLatestPrediction);

// Internal route for other microservices (no auth)
router.get('/internal', getInsights);

module.exports = router;
