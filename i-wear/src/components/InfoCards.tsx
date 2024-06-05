"use client";
import { MdFace3 } from "react-icons/md";
import { GiSunglasses } from "react-icons/gi";
import { FaRegChartBar } from "react-icons/fa";

const cardsData = [
  {
    title: "Facial Information",
    icon: <MdFace3 className="text-6xl" />,
    description: "Details to empower your choice",
  },
  {
    title: "Glasses Suggestions",
    icon: <GiSunglasses className="text-6xl" />,
    description: "Oval to Square, they're all here",
  },
  {
    title: "Match Likelihoods",
    icon: <FaRegChartBar className="text-6xl" />,
    description: "Finding the right one for you",
  },
];

export default function InfoCards() {
  return (
    <div className="bg-white rounded-b-lg border-black p-4 flex justify-center w-fit h-fit">
      <div className="flex flex-col gap-6 items-center">
        <div className="flex flex-col gap-1 items-center">
          <h1 className="text-xl font-extrabold">
            Eyewear catered just for you.
          </h1>
          <p className="text-sm">
            We provide you with the best options based on your facial structure.
          </p>
        </div>
        <ul className="flex flex-row gap-8">
          {cardsData.map((card, index) => (
            <li
              key={index}
              className="flex flex-col items-center gap-1 min-w-60"
            >
              {card.icon}
              <h2 className="text-lg font-bold">{card.title}</h2>
              <p className="text-sm">{card.description}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
