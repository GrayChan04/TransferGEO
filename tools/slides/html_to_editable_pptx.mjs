import pptxgen from "pptxgenjs";
import { chromium } from "playwright";
import fs from "fs/promises";
import path from "path";

const args = process.argv.slice(2);
function arg(name, def=null){ const i=args.indexOf(`--${name}`); return i>=0 && i+1<args.length ? args[i+1] : def; }

const input=arg("input");
const output=arg("output","slides/pptx/deck.pptx");
const selector=arg("selector",".slide");
if(!input){ console.error("Missing --input"); process.exit(1); }

const PPT_W=13.333, PPT_H=7.5;
function color(css){ const m=String(css||"").match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/); return m ? [m[1],m[2],m[3]].map(v=>(+v).toString(16).padStart(2,"0")).join("").toUpperCase() : undefined; }
function pt(px){ const n=Number(String(px||"").replace("px","")); return Number.isFinite(n)? n*.75 : undefined; }

const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720}});
await page.goto(`file://${path.resolve(input)}`,{waitUntil:"networkidle"});

const data=await page.evaluate((selector)=>{
  function rect(el){ const r=el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }
  function style(el){ const s=getComputedStyle(el); return {color:s.color,bg:s.backgroundColor,fontSize:s.fontSize,fontFamily:s.fontFamily,fontWeight:s.fontWeight,textAlign:s.textAlign}; }
  return [...document.querySelectorAll(selector)].map(slide=>{
    const sr=rect(slide), ss=style(slide);
    const els=[...slide.querySelectorAll("h1,h2,h3,p,li,[data-ppt-text],img,table")].map(el=>({
      tag:el.tagName.toLowerCase(), text:(el.innerText||el.alt||"").trim(), rect:rect(el), style:style(el), src:el.currentSrc||el.src||"", html:el.outerHTML
    }));
    return {rect:sr, style:ss, els};
  });
}, selector);

const pptx=new pptxgen(); pptx.layout="LAYOUT_WIDE";
await fs.mkdir(path.dirname(output),{recursive:true});

for(const sd of data){
  const slide=pptx.addSlide();
  const bg=color(sd.style.bg); if(bg) slide.background={color:bg};
  for(const el of sd.els){
    const x=(el.rect.x-sd.rect.x)/sd.rect.w*PPT_W, y=(el.rect.y-sd.rect.y)/sd.rect.h*PPT_H, w=el.rect.w/sd.rect.w*PPT_W, h=el.rect.h/sd.rect.h*PPT_H;
    if(el.tag==="img"){
      try {
        const p = el.src.startsWith("file://") ? new URL(el.src).pathname : path.resolve(path.dirname(input), el.src);
        slide.addImage({path:p,x,y,w,h});
      } catch {}
    } else {
      slide.addText(el.text || "", {x,y,w,h,fontSize:pt(el.style.fontSize)||16,color:color(el.style.color)||"000000",fontFace:(el.style.fontFamily||"Arial").split(",")[0].replace(/['"]/g,""),bold:Number(el.style.fontWeight)>=600||/^h/.test(el.tag),bullet:el.tag==="li"?{type:"ul"}:undefined,fit:"shrink"});
    }
  }
}
await browser.close();
await pptx.writeFile({fileName:output});
console.log(`Wrote editable PPTX: ${output}`);
