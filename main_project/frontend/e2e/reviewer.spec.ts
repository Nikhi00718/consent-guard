import { expect, test } from "@playwright/test";

const tinyPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAQAAAACACAIAAABr1yBdAAABb0lEQVR4nO3TQQEAEADAQAQXQQJhxfDYXYJ9Nve5A6rW7wD4yQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSHvBtAKoifVeZgAAAABJRU5ErkJggg==",
  "base64",
);

async function upload(page: import("@playwright/test").Page) {
  await page.locator('input[type="file"]').setInputFiles({
    name: "staged-review.png",
    mimeType: "image/png",
    buffer: tinyPng,
  });
  await expect(page.getByText("staged-review.png")).toBeVisible();
}

test("one click erases the detected regions and offers the file", async ({ page }) => {
  test.setTimeout(120_000);
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));

  await page.goto("/");
  await expect(page).toHaveTitle("ConsentGuard reviewer");
  await expect(page.getByRole("heading", { name: /Erase the private parts/i })).toBeVisible();
  await expect(page.getByText(/It will miss things/i)).toBeVisible();
  await expect(page.locator("vite-error-overlay, .vite-error-overlay")).toHaveCount(0);

  await upload(page);
  await page.screenshot({ path: "../../outputs/consentguard-home.png", fullPage: true });
  await page.getByRole("button", { name: /Erase and save/i }).click();

  await expect(page.getByRole("heading", { name: "Your redacted image" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("heading", { name: "Your redacted image" })).toBeFocused();
  const download = page.getByRole("link", { name: /Download the clean image/i });
  await expect(download).toBeVisible();
  await expect(download).toHaveAttribute("download", "consentguard-redacted.png");
  await expect(page.getByText(/Detection settings are experimental/i)).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: "../../outputs/consentguard-verification.png", fullPage: true });
  expect(consoleErrors).toEqual([]);
});

test("the reviewer can correct the mask before saving", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  await upload(page);
  await page.getByRole("button", { name: /Check it myself first/i }).click();

  await expect(page.getByRole("heading", { name: /Correct the redaction boundary/i })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("Preparing native-resolution canvas")).toHaveCount(0, { timeout: 10_000 });
  await expect(page.getByTitle("Brush")).toBeVisible();
  await expect(page.getByText(/Known weak here/i)).toBeVisible();
  await page.screenshot({ path: "../../outputs/consentguard-review.png", fullPage: true });

  await page.getByRole("button", { name: /Save my version/i }).click();
  await expect(page.getByRole("heading", { name: "Your redacted image" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("link", { name: /Download the clean image/i })).toBeVisible();
});

test("upload workspace remains usable on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Choose image" })).toBeVisible();
  await expect(page.getByText("local runtime")).toBeHidden();
  await page.screenshot({ path: "../../outputs/consentguard-mobile.png", fullPage: true });
});
