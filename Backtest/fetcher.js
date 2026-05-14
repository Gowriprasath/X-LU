const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');

// Enable the stealth plugin
puppeteer.use(StealthPlugin());

// Utility function for your 3.5s delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
    // 1. Launch the browser
    // Tip: Keep headless: false for the first run so you can visually confirm 
    // Cloudflare is being bypassed. Switch to 'new' or true later.
    const browser = await puppeteer.launch({
        headless: false,
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1280,800']
    });

    const page = await browser.newPage();

    // Set a realistic viewport to avoid bot detection
    await page.setViewport({ width: 1280, height: 800 });

    console.log('Navigating to main page to clear Cloudflare challenge...');

    // 2. Establish the session
    // Go to the home page first and wait for the network to quiet down
    await page.goto('https://www.forexfactory.com/', { waitUntil: 'networkidle2' });

    // Wait 5 seconds to ensure any hidden Turnstile/JS challenges are solved
    // and the cf_clearance cookie is securely stored in this page instance.
    await delay(5000);

    // 3. Define the weeks (Example subset for testing)
    // ForexFactory format: 3-letter month, day, 4-digit year
    const weeksToFetch = [
        'jan1.2017',
        'jan8.2017',
        'jan15.2017'
    ];

    console.log(`Starting fetch loop for ${weeksToFetch.length} weeks...`);

    // 4. Loop through the targets
    for (let i = 0; i < weeksToFetch.length; i++) {
        const week = weeksToFetch[i];
        const url = `https://www.forexfactory.com/calendar?week=${week}`;

        console.log(`[${i + 1}/${weeksToFetch.length}] Fetching ${week}...`);

        // domcontentloaded is faster than networkidle2 and sufficient for HTML parsing
        await page.goto(url, { waitUntil: 'domcontentloaded' });

        // 5. Extract the data
        const events = await page.evaluate(() => {
            const rows = document.querySelectorAll('tr.calendar__row');
            let weekData = [];

            rows.forEach(row => {
                // ForexFactory's DOM is notoriously nested; these are the primary selectors
                const currency = row.querySelector('.calendar__currency')?.innerText.trim();
                const eventName = row.querySelector('.calendar__event')?.innerText.trim();

                // Impact is usually stored in a span class (e.g., 'icon--ff-impact-red')
                const impactElement = row.querySelector('.calendar__impact span');
                const impact = impactElement ? impactElement.className : null;

                // Only push rows that actually contain event data (skipping date headers)
                if (currency && eventName) {
                    weekData.push({ currency, eventName, impact });
                }
            });
            return weekData;
        });

        console.log(`  ✅  ${events.length} events found.`);

        // Maintain your 3.5s delay to respect rate limits
        await delay(3500);
    }

    // Clean up
    await browser.close();
    console.log('Fetch complete.');
})();