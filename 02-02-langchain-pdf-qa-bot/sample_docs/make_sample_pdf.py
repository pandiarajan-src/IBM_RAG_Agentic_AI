"""Regenerates acme_broadband_handbook.pdf. Optional - the PDF is already in this folder.

    pip install reportlab      (BSD licence; not needed to run the lab)
    python sample_docs/make_sample_pdf.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

SECTIONS = [
    ("1. Plans and pricing", [
        "Acme Broadband offers three home internet plans. The Starter plan delivers download speeds of up to 100 Mbps and costs 29 dollars per month. The Family plan delivers up to 500 Mbps and costs 49 dollars per month. The Gigabit plan delivers up to 1000 Mbps and costs 79 dollars per month.",
        "Every plan includes unlimited data, so there are no overage charges. Prices are fixed for the first 24 months of service. After 24 months the monthly price may rise by no more than 5 dollars, and customers are told in writing at least 60 days before any change.",
        "Customers may move to a faster plan at any time and the new speed is active within one hour. Moving to a slower plan is allowed once every 90 days. There is no charge for changing plans.",
        "Students with a valid student ID receive a 10 percent discount on the Starter and Family plans. The student discount cannot be combined with other promotions.",
    ]),
    ("2. Installation and equipment", [
        "Standard installation is free and is normally scheduled within 5 business days of placing an order. The technician visit takes about two hours and an adult aged 18 or over must be present for the whole appointment.",
        "Each account receives one Acme Wave 6 Wi-Fi router on loan at no extra cost. The router remains the property of Acme Broadband and must be returned within 14 days of cancelling service. If the router is not returned, an unreturned equipment fee of 120 dollars is charged.",
        "Customers who prefer to use their own router may do so. Acme Broadband support can confirm that the line is working but cannot configure third-party equipment.",
        "Mesh extenders for larger homes can be rented for 5 dollars per month each, up to a maximum of three extenders per account.",
    ]),
    ("3. Billing and payments", [
        "Bills are issued on the 1st day of each month and payment is due on the 15th day of the same month. Customers can pay by bank transfer, debit card, credit card or automatic payment.",
        "Customers who enrol in AutoPay receive a discount of 3 dollars on every monthly bill. AutoPay collects the full balance on the due date.",
        "If a bill remains unpaid 10 days after the due date, a late payment fee of 7.50 dollars is added to the account. If a bill remains unpaid 30 days after the due date, service is suspended. Restoring a suspended service costs a reconnection fee of 25 dollars and takes up to 4 hours after payment is received.",
        "Paper bills are available on request for 2 dollars per month. Electronic bills are always free and are kept in the customer portal for 36 months.",
    ]),
    ("4. Troubleshooting your connection", [
        "The light on the front of the Acme Wave 6 router shows the state of the connection. A solid green light means the router is online and working normally. A blinking amber light means the router is starting up or installing a software update, which can take up to 10 minutes; do not unplug the router during this time.",
        "A solid red light means there is no internet signal reaching the router. If the light is red, check that the fibre cable is firmly connected to the port marked WAN, then restart the router.",
        "To restart the router, unplug the power cable, wait 30 seconds, and plug it back in. The router needs about 3 minutes to come back online. Most slow-speed and dropped-connection problems are fixed by a restart.",
        "If the light is still red 15 minutes after a restart, check the service status page for a known outage in your area before contacting support. To reset the router to factory settings, hold the recessed reset button for 10 seconds with a paper clip; this erases the Wi-Fi name and password you chose.",
    ]),
    ("5. Cancellations and refunds", [
        "New customers are covered by a 30-day money-back guarantee. If service is cancelled within 30 days of installation, all monthly charges paid are refunded in full.",
        "After the first 30 days, customers may cancel at any time by giving 30 days notice. There is no contract and no early termination fee. The final bill covers service up to the end of the notice period.",
        "Approved refunds are returned to the original payment method within 7 business days. Refunds cannot be paid in cash or to a different card.",
        "If an outage caused by the Acme Broadband network lasts longer than 24 hours in a row, customers may claim a service credit equal to one day of their monthly charge for each full day of the outage. Credits must be claimed within 60 days of the outage.",
    ]),
    ("6. Contacting support", [
        "Phone support is available on 1-800-555-0142 from 7 am to 10 pm Eastern Time, seven days a week. Live chat in the customer portal is available 24 hours a day.",
        "Emails sent to support@acmebroadband.example are answered within one business day. Please include your account number, which is printed at the top of every bill.",
        "If a problem cannot be fixed remotely, a technician visit is booked free of charge when the fault is with Acme Broadband equipment or the network. If the fault is with customer-owned equipment or home wiring, a visit fee of 45 dollars applies.",
        "Customers who are not satisfied with an answer may ask for the case to be escalated to a support supervisor, who will reply within 2 business days.",
    ]),
]


def build(out: Path) -> None:
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Acme Broadband Customer Handbook", styles["Title"]),
        Paragraph("Fictional sample document for training labs. Version 3.2.", styles["Italic"]),
        Spacer(1, 18),
    ]
    for i, (heading, paragraphs) in enumerate(SECTIONS):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading1"]))
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 10)]
    SimpleDocTemplate(str(out), pagesize=A4, title="Acme Broadband Customer Handbook",
                      author="Lab Course Creator", invariant=True).build(story)


if __name__ == "__main__":
    target = Path(__file__).with_name("acme_broadband_handbook.pdf")
    build(target)
    print(f"Wrote {target}")
