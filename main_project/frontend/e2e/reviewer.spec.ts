import { expect, test } from "@playwright/test";

const tinyPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAQAAAACACAIAAABr1yBdAAABb0lEQVR4nO3TQQEAEADAQAQXQQJhxfDYXYJ9Nve5A6rW7wD4yQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSDMAaQYgzQCkGYA0A5BmANIMQJoBSHvBtAKoifVeZgAAAABJRU5ErkJggg==",
  "base64",
);

test("local reviewer completes the honest blocked-preview flow", async ({ page }) => {
  test.setTimeout(120_000);
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));

  await page.goto("/");
  await expect(page).toHaveTitle("ConsentGuard reviewer");
  await expect(page.getByRole("heading", { name: /Inspect the pixels/i })).toBeVisible();
  await expect(page.locator("vite-error-overlay, .vite-error-overlay")).toHaveCount(0);

  await page.locator('input[type="file"]').setInputFiles({
    name: "staged-review.png",
    mimeType: "image/png",
    buffer: tinyPng,
  });
  await expect(page.getByText("staged-review.png")).toBeVisible();
  await page.getByRole("button", { name: /Run local analysis/i }).click();
  await expect(page.getByRole("heading", { name: /Correct the redaction boundary/i })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("Preparing native-resolution canvas")).toHaveCount(0, { timeout: 10_000 });
  await expect(page.getByTitle("Brush")).toBeVisible();
  await page.screenshot({ path: "../../outputs/consentguard-review.png", fullPage: true });

  await page.getByLabel("Consent state").selectOption("GRANTED");
  await page.getByLabel("Subject reference").fill("subject-browser-fixture");
  await page.getByLabel("Audience").fill("local-review-team");
  await page.getByLabel(/I inspected the full image/i).check();
  await page.getByRole("button", { name: /Render and verify/i }).click();
  await expect(page.getByRole("alert")).toHaveText("Add a release purpose before verification.");
  await expect(page.getByLabel("Purpose")).toBeFocused();

  await page.getByLabel("Purpose").fill("Browser verification fixture");
  await page.getByRole("button", { name: /Render and verify/i }).click();

  await expect(page.getByRole("heading", { name: "Verification result" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Verification result" })).toBeFocused();
  await expect(page.getByRole("button", { name: "Download blocked" })).toBeDisabled();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: "../../outputs/consentguard-verification.png", fullPage: true });
  expect(consoleErrors).toEqual([]);
});

test("upload workspace remains usable on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Choose image" })).toBeVisible();
  await expect(page.getByText("local runtime")).toBeHidden();
  await page.screenshot({ path: "../../outputs/consentguard-mobile.png", fullPage: true });
});
