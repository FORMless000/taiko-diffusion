import type { Metadata } from "next";

import "./globals.css";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

export const metadata: Metadata = {
  title: "Taiko Diffusion Web App",
  description: "Generate osu!taiko beatmaps from audio with taiko-diffusion.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <script src={`${basePath}/config.js`} />
      </head>
      <body>{children}</body>
    </html>
  );
}
