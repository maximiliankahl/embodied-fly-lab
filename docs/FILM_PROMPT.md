# Prompt: Embodied-Fly-Lab-Film nachbauen und optimieren

Diesen Text komplett in Claude Code (oder ein anderes Claude mit Dateizugriff und Terminal) kopieren.

---

Du bist mein Video-Engineer. Ziel ist ein **hochwertiger Produktfilm im Stil eines Apple-Produktvideos** (ca. 50 bis 60 s, 1920×1080, später gern 60 fps oder 4K) für unser Hackathon-Projekt **Embodied Fly Lab** (Hack-Nation 7, Challenge 03 „Agentic Scientific Discovery“, Databricks/Omnigent). Es gibt schon eine funktionierende Version (53 s, 30 fps). **Baue sie lokal nach, prüfe sie und verbessere sie.**

## 1. Projekt holen und Version 1 reproduzieren
```bash
git clone https://github.com/maximiliankahl/embodied-fly-lab.git
cd embodied-fly-lab
python -m http.server 8777 --directory web          # in einem eigenen Terminal laufen lassen
```
- Vorschau in Echtzeit (Endlosschleife): http://localhost:8777/film.html
- Deterministischer Modus für das Rendern: http://localhost:8777/film.html?capture. `window.__film.renderAt(t)` zeichnet genau das Bild zur Sekunde t.
- Einzelbilder zum Prüfen: `uv run --with playwright python spikes/film/render_film.py --preview 3,10,17,23,26.5,31,36,40.5,44.5,50`
- Alle Bilder: `uv run --with playwright python spikes/film/render_film.py --fps 30` (Ausgabe in `spikes/film/out/frames/`)
- Browser-Wahl im Skript: `--channel msedge` (Windows), `--channel chrome` oder `--channel chromium` (vorher `uv run --with playwright python -m playwright install chromium`).
- MP4 erzeugen:
  `ffmpeg -framerate 30 -i spikes/film/out/frames/%05d.jpg -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -movflags +faststart film.mp4`

**Die relevanten Dateien:**
- `web/film.html`: Bühne, Typografie, Overlays
- `web/js/film.js`: Szene, Kamera, Zeitplan, alles in Three.js 0.169 per importmap von jsDelivr
- `spikes/film/render_film.py`: Bild-für-Bild-Renderer mit Playwright
- Daten, alles echte, exportierte Simulationen:
  - `web/data/brain_points.json/.bin`: 138.639 Neuronen-Positionen aus FlyWire v783, in µm
  - `web/data/geometry/flybody.json/.bin` (Einheit cm) und `neuromechfly.json/.bin` (Einheit mm): Körpergeometrie
  - `web/data/runs/*.json`: Läufe mit Posen (`body.poses`: `t`, `p`, `q` pro Körperteil, Quaternionen w,x,y,z), aktiven Neuronen (`brain.active_idx`, `active_rate_hz`, `stimulated_idx`) und Prüfurteilen

## 2. Storyboard (Version 1, bitte beibehalten und verfeinern)
| Zeit | Bild | Text (Englisch) |
|---|---|---|
| 0–7 s | Nur das Gehirn, 138.639 Punkte, langsame Drehung, schwarzer Hintergrund | „138,639 neurons.“ / „The complete wiring of a fruit-fly brain.“ |
| 7–14 s | Kamera fährt zurück, gläserne Fliege (FlyBody) erscheint, Gehirn leuchtet im Kopf | „Now it has a body.“ |
| 14–20 s | Leuchtimpulse vom Gehirn zu Beinen und Flügeln | „Neurons, wired to muscles.“ |
| 20–28 s | Lauf `story_miswired_dna02l` (NeuroMechFly): absichtlich falsch verdrahteter Adapter, die Fliege dreht nach rechts statt links, Impulse rot | „Test 1: steering.“ → „Wrong turn.“ |
| 28–34 s | Bewegung läuft rückwärts („rückgängig“), dann Lauf `dna02l_turn_left` in Grün | „Detected. Rewired.“ → „Verified.“ |
| 34–43 s | Lauf `gf_dng02_climb` (FlyBody): Giant Fiber und DNg02 leuchten, Abheben, Steigflug mit Leuchtspur | „Test 2: escape.“ → „Lift-off.“ |
| 43–47 s | Zuckerwürfel, Zucker-Neuronen leuchten (Aktivität aus `story_sugar_feeding`) | „Taste.“ / „Sugar neurons light up the feeding neuron MN9.“ |
| 47–53 s | Schlusskarte | „Embodied Fly Lab“ + Leitsatz + Credits/Repo-Link |

Leitsatz: *„Our agents solve complex scientific problems by testing, validating through simulation and comparing with published research.“*

## 3. Harte Regeln (wissenschaftliche Ehrlichkeit, Teamregel G1)
- Jede Körperbewegung kommt **nur aus aufgezeichneten Posen** in `web/data/runs/*.json`. Keine von Hand animierten Bein- oder Flügelbewegungen, kein Keyframing des Körpers.
- Leuchtende Neuronen **nur aus aufgezeichneter Aktivität** (`brain.active_idx`, `active_rate_hz`). Die Daten sind mittlere Raten über 1 s, nicht zeitaufgelöst. Ein zeitlicher Verlauf darf nur als Ein- oder Ausblenden dargestellt werden, nicht als erfundene Spike-Abfolge.
- Kennzeichnung im Bild (kleine Fußzeile) muss bleiben:
  - „Recorded simulations“
  - „Brain shown scaled inside the head“ (das echte Gehirn wird für die Darstellung in den Kopf skaliert)
  - „Sugar cube is an illustration“ (der Zielflug zum Zucker ist **nicht** simuliert)
- Erlaubte Zahlen:
  - 138,639 Neuronen
  - Gehirnmodell gegen das Original: r = 0.999
  - 8 von 9 vergleichbaren Literatur-Checks stimmen
  - 25× weniger Simulationen bis zum ersten Treffer gegen zufällige Reihenfolge, 5× gegen eine starke Vergleichsmethode
  - Keine anderen Zahlen erfinden.
- Der Fehlversuch ist ein **absichtlicher Negativ-Kontrollversuch** (Adapter v0 mit vertauschtem Vorzeichen). Er darf nicht als Zufallsfehler dargestellt werden.

## 4. Bekannte Fallstricke (schon gelöst, nicht wieder einbauen)
- **Koordinaten:** MuJoCo ist z-oben, Three.js y-oben, deshalb `world.rotation.x = -π/2`. Posen von FlyBody sind in cm (×10 → mm), NeuroMechFly in mm.
- **Punktgröße** mit `sizeAttenuation` in Welt-Einheiten (mm). Gehirnpunkte etwa 0.0035, nicht 2. Zu große Punkte führen zu GPU-Überlast und „WebGL context lost“.
- **Fresnel-Glas-Shader:** Normalen und Blickvektor gegen Länge 0 absichern und `pow(clamp(1-|n·v|,0,1), k)` verwenden. Sonst entstehen NaN-Pixel, und der Bloom-Pass färbt ganze Bilder schwarz.
- **Gehirn ausrichten:** Der FlyBody-Körperteil `head` ist um 90° gedreht. Das Gehirn deshalb an der Thorax-Quaternion ausrichten, nur die Position vom Kopf nehmen. Bei NeuroMechFly gibt es keinen Kopf-Körper: Mitte zwischen `nmf/l_eye` und `nmf/r_eye` verwenden.
- **Flug-Posen** sind stroboskopisch (ca. 65 Posen für 1 s Simulation). Interpolieren; die Kamera folgt dem Thorax und darf nicht zum Zucker wegschwenken, solange die Fliege fliegt.
- **Abblende** (`#fade`) muss unter dem Text liegen (z-index), sonst ist die Schlusskarte schwarz.
- **Kein `MeshPhysicalMaterial` mit `transmission`** im Zusammenspiel mit Bloom (Risiko schwarzer Bilder). Lieber `MeshStandardMaterial` mit leichter Emission.

## 5. Bitte optimieren (Vorschläge, nach Wirkung sortiert)
1. **Kamera:** weiche Spline-Kamerafahrten (CatmullRom) statt linearer Orbits, leichte Tiefenunschärfe (BokehPass) bei Nahaufnahmen, 60 fps.
2. **Gehirn-zu-Körper-Moment:** Die Fliege sollte sich um das Gehirn herum „aufbauen“, z. B. Körperteile nacheinander einblenden. Dazu Leuchtbahnen entlang des Halses statt gerader Kurven.
3. **Rückgängig-Effekt** bei 28 s: kurzer „Rewind“-Look (Zeitlupe rückwärts, leichter Farbversatz), dann grüner Neustart.
4. **Abheben:** Kamera unter die Fliege legen und mit ihr steigen, Leuchtspur mit Verlauf. Zucker als glänzender Tropfen oder Kristall statt Würfel, weiterhin als Illustration gekennzeichnet.
5. **Typografie:** Schlusskarte vertikal zentriert; Texte im Apple-Stil kurz und knapp, große Ziffern.
6. **Ton:** lizenzfreie Musik (z. B. aus der Clipchamp-Bibliothek) und Sprechertext auf Englisch. Erst danach den Schnitt auf die Musik anpassen.
7. **Varianten:**
   - (a) Produkt-Demo höchstens 60 s
   - (b) Anfang der 2-Minuten-Challenge-Demo, danach Bildschirmaufnahmen vom Streamlit-Dashboard (`uv run streamlit run app.py`, Seite „Lab notebook“) und von der Omnigent-Weboberfläche (Agenten-Übergaben, Freigabe-Karte)

## 6. Vorgehen
Zuerst Version 1 rendern und 10 Standbilder als Kontaktabzug anschauen. Dann Änderungen in kleinen Schritten, nach jeder Änderung wieder Standbilder prüfen, erst am Ende komplett rendern.

Abgabeformat: MP4, H.264, yuv420p, 1920×1080, höchstens 60 s für die Plattform-Videos, unter 1 GB (Ziel unter 50 MB).
