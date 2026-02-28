import "./globals.css";

export const metadata = {
  title: "HandiTalk Demo",
  description: "Demo de reconnaissance gestes",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
