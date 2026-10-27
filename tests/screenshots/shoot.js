// Screenshots of the interface for the README: desktop and phone, light and dark.
//   python3 tests/screenshots/serve_demo.py &   # prints the URL
//   node tests/screenshots/shoot.js http://127.0.0.1:PORT/ docs/
const { chromium } = require('playwright');

(async () => {
  const [url, out = 'docs/'] = process.argv.slice(2);
  const browser = await chromium.launch();
  const shots = [
    ['desktop', 'light', { width: 1180, height: 1500 }, 1],
    ['desktop', 'dark', { width: 1180, height: 1500 }, 1],
    ['mobile', 'light', { width: 390, height: 844 }, 3],
    ['mobile', 'dark', { width: 390, height: 844 }, 3],
  ];
  for (const [kind, scheme, viewport, scale] of shots) {
    const page = await browser.newPage({ viewport, deviceScaleFactor: scale, colorScheme: scheme,
                                         isMobile: kind === 'mobile', hasTouch: kind === 'mobile' });
    await page.goto(url);
    await page.waitForSelector('.rec');
    await page.evaluate(() => document.querySelector('details.run')?.setAttribute('open', ''));
    await page.waitForTimeout(300);
    const file = `${out}modelwatch-${kind}-${scheme}.png`;
    await page.screenshot({ path: file, fullPage: kind === 'desktop' });
    console.log(file);
    await page.close();
  }
  await browser.close();
})();
